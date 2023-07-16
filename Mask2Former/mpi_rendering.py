import torch
import numpy as np
import cv2,os,time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import torch.nn.functional as F
from torch import nn

# from utils import inverse
# from operations.homography_sampler import HomographySample
# from operations.rendering_utils import transform_G_xyz, sample_pdf, gather_pixel_by_pxpy

def transform_G_xyz(G, xyz, is_return_homo=False):
    """

    :param G: Bx4x4
    :param xyz: Bx3xN
    :return:
    """
    assert len(G.size()) == len(xyz.size())
    if len(G.size()) == 2:
        G_B44 = G.unsqueeze(0)
        xyz_B3N = xyz.unsqueeze(0)
    else:
        G_B44 = G
        xyz_B3N = xyz
    xyz_B4N = torch.cat((xyz_B3N, torch.ones_like(xyz_B3N[:, 0:1, :])), dim=1)
    G_xyz_B4N = torch.matmul(G_B44.to(xyz_B4N), xyz_B4N)
    if is_return_homo:
        return G_xyz_B4N
    else:
        return G_xyz_B4N[:, 0:3, :]

def inverse(matrices,src_addr=None,tgt_addr=None):
    """
    torch.inverse() sometimes produces outputs with nan the when batch size is 2.
    Ref https://github.com/pytorch/pytorch/issues/47272
    this function keeps inversing the matrix until successful or maximum tries is reached
    :param matrices Bx3x3
    """
    if (torch.isnan(matrices)).any():
        print('zmf matrices contains nan !!!')
        print(src_addr,tgt_addr)
        print('over')
        # raise Exception("Matrix inverse contains nan!")
        exit()
        
    inverse = None
    max_tries = 5
    while (inverse is None) or (torch.isnan(inverse)).any():
        torch.cuda.synchronize()
        inverse = torch.inverse(matrices)

        # Break out of the loop when the inverse is successful or there"re no more tries
        max_tries -= 1
        if max_tries == 0:
            break

    # Raise an Exception if the inverse contains nan
    if (torch.isnan(inverse)).any():
        print('zmf inverse contains nan !!!')
        print(matrices)
        print(inverse)
        print(src_addr,tgt_addr)
        print('over')
        # raise Exception("Matrix inverse contains nan!")
        exit()



    return inverse

def sample_pdf(values, weights, N_samples):
    """
    draw samples from distribution approximated by values and weights.
    the probability distribution can be denoted as weights = p(values)
    :param values: Bx1xNxS
    :param weights: Bx1xNxS
    :param N_samples: number of sample to draw
    :return:
    """
    B, N, S = weights.size(0), weights.size(2), weights.size(3)
    assert values.size() == (B, 1, N, S)

    # convert values to bin edges
    bin_edges = (values[:, :, :, 1:] + values[:, :, :, :-1]) * 0.5  # Bx1xNxS-1
    bin_edges = torch.cat((values[:, :, :, 0:1],
                           bin_edges,
                           values[:, :, :, -1:]), dim=3)  # Bx1xNxS+1

    pdf = weights / (torch.sum(weights, dim=3, keepdim=True) + 1e-5)  # Bx1xNxS
    cdf = torch.cumsum(pdf, dim=3)  # Bx1xNxS
    cdf = torch.cat((torch.zeros((B, 1, N, 1), dtype=cdf.dtype, device=cdf.device),
                     cdf), dim=3)  # Bx1xNxS+1

    # uniform sample over the cdf values
    u = torch.rand((B, 1, N, N_samples), dtype=weights.dtype, device=weights.device)  # Bx1xNxN_samples

    # get the index on the cdf array
    cdf_idx = torch.searchsorted(cdf, u, right=True)  # Bx1xNxN_samples
    cdf_idx_lower = torch.clamp(cdf_idx-1, min=0)  # Bx1xNxN_samples
    cdf_idx_upper = torch.clamp(cdf_idx, max=S)  # Bx1xNxN_samples

    # linear approximation for each bin
    cdf_idx_lower_upper = torch.cat((cdf_idx_lower, cdf_idx_upper), dim=3)  # Bx1xNx(N_samplesx2)
    cdf_bounds_N2 = torch.gather(cdf, index=cdf_idx_lower_upper, dim=3)  # Bx1xNx(N_samplesx2)
    cdf_bounds = torch.stack((cdf_bounds_N2[..., 0:N_samples], cdf_bounds_N2[..., N_samples:]), dim=4)
    bin_bounds_N2 = torch.gather(bin_edges, index=cdf_idx_lower_upper, dim=3)  # Bx1xNx(N_samplesx2)
    bin_bounds = torch.stack((bin_bounds_N2[..., 0:N_samples], bin_bounds_N2[..., N_samples:]), dim=4)

    # avoid zero cdf_intervals
    cdf_intervals = cdf_bounds[:, :, :, :, 1] - cdf_bounds[:, :, :, :, 0] # Bx1xNxN_samples
    bin_intervals = bin_bounds[:, :, :, :, 1] - bin_bounds[:, :, :, :, 0]  # Bx1xNxN_samples
    u_cdf_lower = u - cdf_bounds[:, :, :, :, 0]  # Bx1xNxN_samples
    # there is the case that cdf_interval = 0, caused by the cdf_idx_lower/upper clamp above, need special handling
    t = u_cdf_lower / torch.clamp(cdf_intervals, min=1e-5)
    t = torch.where(cdf_intervals <= 1e-4,
                    torch.full_like(u_cdf_lower, 0.5),
                    t)

    samples = bin_bounds[:, :, :, :, 0] + t*bin_intervals
    return samples

class HomographySample:
    def __init__(self, H_tgt, W_tgt, device=None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.Height_tgt = H_tgt
        self.Width_tgt = W_tgt
        # self.meshgrid = self.grid_generation(self.Height_tgt, self.Width_tgt, self.device)
        # self.meshgrid = self.meshgrid.permute(2, 0, 1).contiguous()  # 3xHxW
        self.meshgrid2 = self.grid_generation2(self.Height_tgt, self.Width_tgt, self.device)
        self.meshgrid2 = self.meshgrid2.permute(2, 0, 1).contiguous()  # 3xHxW

        self.n = self.plane_normal_generation(self.device)

    @staticmethod
    def grid_generation(H, W, device):
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xv, yv = np.meshgrid(x, y)  # HxW
        xv = torch.from_numpy(xv.astype(np.float32)).to(dtype=torch.float32, device=device)
        yv = torch.from_numpy(yv.astype(np.float32)).to(dtype=torch.float32, device=device)
        ones = torch.ones_like(xv)
        meshgrid = torch.stack((xv, yv, ones), dim=2)  # HxWx3

        return meshgrid

    @staticmethod
    def grid_generation2(H, W, device):

        # xy_map = torch.zeros((2, H, W),device=device)
        # for y in range(H):
        #     for x in range(W):
        #         # yy = float(y) / H * 480
        #         # xx = float(x) / W * 640
        #         xy_map[0, y, x] = float(x) / W * 640
        #         xy_map[1, y, x] = float(y) / H * 480
        # xy_map = xy_map.permute(1,2,0)
        # ones = torch.ones_like(xy_map[:,:,0:1])
        # meshgrid2 = torch.cat([xy_map,ones],dim=2)

        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xv, yv = np.meshgrid(x, y)  # HxW
        xv = torch.from_numpy(xv.astype(np.float32)).to(dtype=torch.float32, device=device) / W * 640
        yv = torch.from_numpy(yv.astype(np.float32)).to(dtype=torch.float32, device=device) / H * 480
        ones = torch.ones_like(xv)
        meshgrid2 = torch.stack([xv, yv,ones],dim=2)

        return meshgrid2

    @staticmethod
    def plane_normal_generation(device):
        n = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        return n

    @staticmethod
    def euler_to_rotation_matrix(x_angle, y_angle, z_angle, seq='xyz', degrees=False):
        """
        Note that here we want to return a rotation matrix rot_mtx, which transform the tgt points into src frame,
        i.e, rot_mtx * p_tgt = p_src
        Therefore we need to add negative to x/y/z_angle
        :param roll:
        :param pitch:
        :param yaw:
        :return:
        """
        r = Rotation.from_euler(seq,
                                [-x_angle, -y_angle, -z_angle],
                                degrees=degrees)
        rot_mtx = r.as_matrix().astype(np.float32)
        return rot_mtx


    def sample(self, src_view_norm_para_BS3, src_BCHW, d_src_B,
               G_tgt_src,
               K_src_inv, K_tgt, src_addr=None,tgt_addr=None):
        """
        Coordinate system: x, y are the image directions, z is pointing to depth direction
        :param src_BCHW: torch tensor float, 0-1, rgb/rgba. BxCxHxW
                         Assume to be at position P=[I|0]
        :param d_src_B: distance of image plane to src camera origin
        :param G_tgt_src: Bx4x4
        :param K_src_inv: Bx3x3
        :param K_tgt: Bx3x3
        :return: tgt_BCHW
        """
        # parameter processing ------ begin ------
        B, channels, Height_src, Width_src = src_BCHW.size(0), src_BCHW.size(1), src_BCHW.size(2), src_BCHW.size(3)

        R_tgt_src = G_tgt_src[:, 0:3, 0:3]
        t_tgt_src = G_tgt_src[:, 0:3, 3]

        K_640 = torch.tensor([[577.,   0., 320.], [  0., 577., 240.], [  0.,   0.,   1.]])[None,...].repeat(B,1,1).to(device=src_BCHW.device)
        K_640_inv = torch.inverse(K_640)

        # if R_src_tgt is None:
        #     R_src_tgt = torch.eye(3, dtype=torch.float32, device=src_BCHW.device)
        #     R_src_tgt = R_src_tgt.unsqueeze(0).expand(B, 3, 3)
        # if t_src_tgt is None:
        #     t_src_tgt = torch.tensor([0, 0, 0],
        #                              dtype=torch.float32,
        #                              device=src_BCHW.device)
        #     t_src_tgt = t_src_tgt.unsqueeze(0).expand(B, 3)

        # relationship between FoV and focal length:
        # assume W > H
        # W / 2 = f*tan(\theta / 2)
        # here we default the horizontal FoV as 53.13 degree
        # the vertical FoV can be computed as H/2 = W*tan(\theta/2)

        R_tgt_src = R_tgt_src.to(device=src_BCHW.device)
        t_tgt_src = t_tgt_src.to(device=src_BCHW.device)
        K_src_inv = K_640_inv.to(device=src_BCHW.device) # K_src_inv.to(device=src_BCHW.device)
        K_tgt     = K_640.to(device=src_BCHW.device) # K_tgt.to(device=src_BCHW.device)
        # parameter processing ------ end ------

        # the goal is compute H_src_tgt, that maps a tgt pixel to src pixel
        # so we compute H_tgt_src first, and then inverse
        # n = self.n.to(device=src_BCHW.device)
        # n = n.unsqueeze(0).repeat(B, 1)  # Bx3

        # Bx3x3 - (Bx3x1 * Bx1x3)
        # note here we use -d_src, because the plane function is n^T * X - d_src = 0
        d_src_B33 = d_src_B.reshape(B, 1, 1).repeat(1, 3, 3).to(device=src_BCHW.device)  # B -> Bx3x3

        

        # R_tnd = R_tgt_src - torch.matmul(t_tgt_src.unsqueeze(2), n.unsqueeze(1)) / -d_src_B33
        R_tnd = R_tgt_src.to(t_tgt_src) - torch.matmul(t_tgt_src.unsqueeze(2), src_view_norm_para_BS3.to(t_tgt_src).reshape(B,3).unsqueeze(1)) / -d_src_B33

        if (torch.isinf(R_tnd)).any():
            print('R_tnd inf')
            print(R_tnd)
            exit()

        # print(K_tgt.dtype, R_tnd.dtype, K_src_inv.dtype)
        # print(K_tgt.device, R_tnd.device, K_src_inv.device)
        H_tgt_src = torch.matmul(K_tgt, torch.matmul(R_tnd, K_src_inv))

        # TODO: fix cuda inverse
        with torch.no_grad():
            datatype = H_tgt_src.dtype
            H_src_tgt = inverse(H_tgt_src.float(),src_addr,tgt_addr).to(dtype=datatype)
            # H_src_tgt = torch.from_numpy(np.linalg.inv(H_tgt_src.cpu().numpy())).to(device=H_tgt_src.device, dtype=torch.float32)

        # create tgt image grid, and map to src
        meshgrid_tgt_homo = self.meshgrid2.to(src_BCHW.device)
        # 3xHxW -> Bx3xHxW
        meshgrid_tgt_homo = meshgrid_tgt_homo.unsqueeze(0).expand(B, 3, Height_src, Width_src)

        # wrap meshgrid_tgt_homo to meshgrid_src
        meshgrid_tgt_homo_B3N = meshgrid_tgt_homo.view(B, 3, -1)  # Bx3xHW
        meshgrid_src_homo_B3N = torch.matmul(H_src_tgt.to(meshgrid_tgt_homo_B3N), meshgrid_tgt_homo_B3N)  # Bx3x3 * Bx3xHW -> Bx3xHW

        # Bx3xHW -> Bx3xHxW -> BxHxWx3
        meshgrid_src_homo = meshgrid_src_homo_B3N.view(B, 3, Height_src, Width_src).permute(0, 2, 3, 1)
        meshgrid_src = meshgrid_src_homo[:, :, :, 0:2] / meshgrid_src_homo[:, :, :, 2:]  # BxHxWx2

        valid_mask_x = torch.logical_and(meshgrid_src[:, :, :, 0] < 640,
                                         meshgrid_src[:, :, :, 0] > -1)
        valid_mask_y = torch.logical_and(meshgrid_src[:, :, :, 1] < 480,
                                         meshgrid_src[:, :, :, 1] > -1)
        valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # BxHxW

        # sample from src_BCHW
        # normalize meshgrid_src to [-1,1]
        meshgrid_src[:, :, :, 0] = (meshgrid_src[:, :, :, 0]+0.5) / (640 * 0.5) - 1
        meshgrid_src[:, :, :, 1] = (meshgrid_src[:, :, :, 1]+0.5) / (480 * 0.5) - 1

        if os.environ['BLACK_BORDER'] != 'True':
            tgt_BCHW = torch.nn.functional.grid_sample(src_BCHW, grid=meshgrid_src, padding_mode='border', align_corners=False)
        else:
            tgt_BCHW = torch.nn.functional.grid_sample(src_BCHW, grid=meshgrid_src, padding_mode='zeros', align_corners=False)

        return tgt_BCHW, valid_mask

def render(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, use_alpha=False, is_bg_depth_inf=False):
    if not use_alpha:
        imgs_syn, depth_syn, blend_weights, weights = plane_volume_rendering(
            rgb_BS3HW,
            sigma_BS1HW,
            xyz_BS3HW,
            is_bg_depth_inf
        )
    else:
        imgs_syn, weights = alpha_composition(sigma_BS1HW, rgb_BS3HW)
        depth_syn, _ = alpha_composition(sigma_BS1HW, xyz_BS3HW[:, :, 2:])
        # No rgb blending with alpha composition
        blend_weights = torch.zeros_like(rgb_BS3HW).cuda()
    return imgs_syn, depth_syn, blend_weights, weights

def alpha_composition(alpha_BK1HW, value_BKCHW):
    """
    composition equation from 'Single-View View Synthesis with Multiplane Images'
    K is the number of planes, k=0 means the nearest plane, k=K-1 means the farthest plane
    :param alpha_BK1HW: alpha at each of the K planes
    :param value_BKCHW: rgb/disparity at each of the K planes
    :return:
    """
    B, K, _, H, W = alpha_BK1HW.size()
    alpha_comp_cumprod = torch.cumprod(1 - alpha_BK1HW, dim=1, dtype=alpha_BK1HW.dtype)  # BxKx1xHxW

    preserve_ratio = torch.cat((torch.ones((B, 1, 1, H, W), dtype=alpha_BK1HW.dtype, device=alpha_BK1HW.device),
                                alpha_comp_cumprod[:, 0:K-1, :, :, :]), dim=1)  # BxKx1xHxW
    weights = alpha_BK1HW * preserve_ratio  # BxKx1xHxW
    value_composed = torch.sum(value_BKCHW * weights, dim=1, keepdim=False, dtype=alpha_BK1HW.dtype)  # Bx3xHxW


    alpha_acc = torch.sum(weights, dim=1, keepdim=False, dtype=alpha_BK1HW.dtype)  # Bx1xHxW

    # print(alpha_acc.shape,5678)

    see_alpha = (alpha_acc[0,0]*255).type(torch.uint8).cpu().numpy()
    # import random
    # cv2.imwrite('./zmf_debug/%d_a.png'%(int(random.random()*100)),see_alpha)



    return value_composed, alpha_acc

def plane_volume_rendering(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, is_bg_depth_inf):
    B, S, _, H, W = sigma_BS1HW.size()

    xyz_diff_BS3HW = xyz_BS3HW[:, 1:, :, :, :] - xyz_BS3HW[:, 0:-1, :, :, :]  # Bx(S-1)x3xHxW
    xyz_dist_BS1HW = torch.norm(xyz_diff_BS3HW, dim=2, keepdim=True)  # Bx(S-1)x1xHxW
    xyz_dist_BS1HW = torch.cat((xyz_dist_BS1HW,
                                torch.full((B, 1, 1, H, W),
                                           fill_value=1e3,
                                           dtype=xyz_BS3HW.dtype,
                                           device=xyz_BS3HW.device)),
                               dim=1)  # BxSx3xHxW
    transparency = torch.exp(-sigma_BS1HW * xyz_dist_BS1HW)  # BxSx1xHxW
    alpha = 1 - transparency # BxSx1xHxW

    # add small eps to avoid zero transparency_acc
    # pytorch.cumprod is like: [a, b, c] -> [a, a*b, a*b*c], we need to modify it to [1, a, a*b]
    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)  # BxSx1xHxW
    transparency_acc = torch.cat((torch.ones((B, 1, 1, H, W), dtype=transparency.dtype, device=transparency.device),
                                  transparency_acc[:, 0:-1, :, :, :]),
                                 dim=1)  # BxSx1xHxW

    weights = transparency_acc * alpha  # BxSx1xHxW
    rgb_out, depth_out = weighted_sum_mpi(rgb_BS3HW, xyz_BS3HW, weights, is_bg_depth_inf)

    return rgb_out, depth_out, transparency_acc, weights

def weighted_sum_mpi(rgb_BS3HW, xyz_BS3HW, weights, is_bg_depth_inf):
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # Bx1xHxW
    rgb_out = torch.sum(weights * rgb_BS3HW, dim=1, keepdim=False)  # Bx3xHxW

    if is_bg_depth_inf:
        # for dtu dataset, set large depth if weight_sum is small
        depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False) \
                    + (1 - weights_sum) * 1000
    else:
        depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False) \
                    / (weights_sum + 1e-5)  # Bx1xHxW

    return rgb_out, depth_out

def get_xyz_from_depth(meshgrid_homo,
                       depth,
                       K_inv):
    """

    :param meshgrid_homo: 3xHxW
    :param depth: Bx1xHxW
    :param K_inv: Bx3x3
    :return:
    """
    H, W = meshgrid_homo.size(1), meshgrid_homo.size(2)
    B, _, H_d, W_d = depth.size()
    assert H==H_d, W==W_d

    # 3xHxW -> Bx3xHxW
    meshgrid_src_homo = meshgrid_homo.unsqueeze(0).repeat(B, 1, 1, 1)
    meshgrid_src_homo_B3N = meshgrid_src_homo.reshape(B, 3, -1)
    xyz_src = torch.matmul(K_inv, meshgrid_src_homo_B3N)  # Bx3xHW
    xyz_src = xyz_src.reshape(B, 3, H, W) * depth  # Bx3xHxW

    return xyz_src

def get_src_xyz_from_plane_disparity(meshgrid_src_homo,
                                     mpi_disparity_src,
                                     K_src_inv):
    """
    :param meshgrid_src_homo: 3xHxW
    :param mpi_disparity_src: BxS
    :param K_src_inv: Bx3x3
    :return:
    """
    B, S = mpi_disparity_src.size()
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS
    # print(K_src_inv.size())
    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33.to(meshgrid_src_homo_Bs3N), meshgrid_src_homo_Bs3N)  # BSx3xHW


    xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3).to(xyz_src)  # BxSx3xHW
    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW

# zmf
def get_src_xyz_from_plane_para(meshgrid_src_homo,
                                     para_BS3,
                                     K_src_inv):
    """
    :param meshgrid_src_homo: 3xHxW
    :param mpi_disparity_src: BxS
    :param K_src_inv: Bx3x3
    :return:
    """
    B, S = para_BS3.size()[0:2]
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    # mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS
    # print(K_src_inv.size())
    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33.to(meshgrid_src_homo_Bs3N), meshgrid_src_homo_Bs3N)  # BSx3xHW



    depth_maps_inv = torch.matmul(para_BS3.reshape(B * S, 3), xyz_src[0,:,:]) # BSx3 x 3xHW = BSxHW
    depth_maps_inv = torch.clamp(depth_maps_inv, min=0.001, max=1e4) 
    depth_maps = 1. / depth_maps_inv

    # xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3).to(xyz_src)  # BxSx3xHW
    # xyz_src = xyz_src.reshape(B, S, 3, H * W) * para_BS3.unsqueeze(3).to(xyz_src)  # BxSx3xHW


    xyz_src = xyz_src.reshape(B, S, 3, H * W) * depth_maps.reshape(B, S, 1, H * W).repeat(1,1,3,1)  # BxSx3xHW

    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW

def get_tgt_xyz_from_plane_disparity(xyz_src_BS3HW,
                                     G_tgt_src):
    """

    :param xyz_src_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :return:
    """
    B, S, _, H, W = xyz_src_BS3HW.size()
    G_tgt_src_Bs33 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).reshape(B*S, 4, 4)
    xyz_tgt = transform_G_xyz(G_tgt_src_Bs33, xyz_src_BS3HW.reshape(B*S, 3, H*W))  # Bsx3xHW
    xyz_tgt_BS3HW = xyz_tgt.reshape(B, S, 3, H, W)  # BxSx3xHxW
    return xyz_tgt_BS3HW

def render_tgt_rgb_depth_mv(H_sampler: HomographySample,
                         all_mpi_rgb_src,
                         all_mpi_sigma_src,
                         all_mpi_disparity_src,
                         all_xyz_tgt_BS3HW,
                         all_G_tgt_src,
                         all_K_src_inv, K_tgt,
                         use_alpha=False,
                         is_bg_depth_inf=False):
    """
    :param H_sampler:
    :param mpi_rgb_src: BxSx3xHxW
    :param mpi_sigma_src: BxSx1xHxW
    :param mpi_disparity_src: BxS
    :param xyz_tgt_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :param K_src_inv: Bx3x3
    :param K_tgt: Bx3x3
    :return:
    """
    B, S, _, H, W = all_mpi_rgb_src[0].size()
    tgt_mpi_xyz_mv = torch.empty(B, S, 7, H, W)
    tgt_mask_mv = torch.ones(B, 1, H, W)
    for i in range(len(all_mpi_rgb_src)):
        mpi_rgb_src = all_mpi_rgb_src[i]
        mpi_disparity_src = all_mpi_disparity_src[i]
        mpi_sigma_src = all_mpi_sigma_src[i]
        xyz_tgt_BS3HW = all_xyz_tgt_BS3HW[i]
        G_tgt_src = all_G_tgt_src[i]
        K_src_inv = all_K_src_inv[i]
        B, S, _, H, W = mpi_rgb_src.size()

        mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

        # note that here we concat the mpi_src with xyz_tgt, because H_sampler will sample them for tgt frame
        # mpi_src is the same in whatever frame, but xyz has to be in tgt frame
        mpi_xyz_src = torch.cat((mpi_rgb_src, mpi_sigma_src, xyz_tgt_BS3HW), dim=2)  # BxSx(3+1+3)xHxW

        # homography warping of mpi_src into tgt frame
        G_tgt_src_Bs44 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 4, 4)  # Bsx4x4
        K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
        K_tgt_Bs33 = K_tgt.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3

        # BsxCxHxW, BsxHxW
        tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = H_sampler.sample(mpi_xyz_src.view(B*S, 7, H, W),
                                                            mpi_depth_src.view(B*S),
                                                            G_tgt_src_Bs44,
                                                            K_src_inv_Bs33,
                                                            K_tgt_Bs33)

        # mpi composition
        tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 7, H, W)
        tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
        tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
                                torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device),
                                torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device))
        tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW
        if (i==0):
            tgt_mpi_xyz_mv =torch.unsqueeze(tgt_mpi_xyz,dim=1)
            tgt_mask_mv = tgt_mask
        else:
            tgt_mpi_xyz_mv = torch.cat([tgt_mpi_xyz_mv,torch.unsqueeze(tgt_mpi_xyz,dim=1)],dim=1)
            tgt_mask_mv = torch.logical_and(tgt_mask_mv,tgt_mask)

    
    tgt_rgb_BS3HW = tgt_mpi_xyz_mv[:,:, :, 0:3, :, :].mean(dim=1,keepdim=False)
    tgt_sigma_BS1HW = tgt_mpi_xyz_mv[:,:, :, 3:4, :, :].mean(dim=1,keepdim=False)
    # To DO Change this to another better implementation
    tgt_xyz_BS3HW = tgt_mpi_xyz_mv[:,:, :, 4:, :, :].mean(dim=1,keepdim=False)

    # tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
    # tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
    #                             torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device),
    #                             torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device))

    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))
    tgt_rgb_syn, tgt_depth_syn, _, _ = render(tgt_rgb_BS3HW, tgt_sigma_BS1HW, tgt_xyz_BS3HW,
                                              use_alpha=use_alpha,
                                              is_bg_depth_inf=is_bg_depth_inf)
    # tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW

    return tgt_rgb_syn, tgt_depth_syn, tgt_mask_mv

def render_tgt_rgb_depth_mv_llff(H_sampler: HomographySample,
                         all_mpi_rgb_src,
                         all_mpi_sigma_src,
                         all_mpi_disparity_src,
                         all_xyz_tgt_BS3HW,
                         all_G_tgt_src,
                         all_K_src_inv, K_tgt,
                         use_alpha=False,
                         is_bg_depth_inf=False):
    """
    :param H_sampler:
    :param mpi_rgb_src: BxSx3xHxW
    :param mpi_sigma_src: BxSx1xHxW
    :param mpi_disparity_src: BxS
    :param xyz_tgt_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :param K_src_inv: Bx3x3
    :param K_tgt: Bx3x3
    :return:
    """
    B, S, _, H, W = all_mpi_rgb_src[0].size()
    tgt_mpi_xyz_mv = torch.empty(B, S, 7, H, W)
    tgt_mask_mv = torch.ones(B, 1, H, W)
    for i in range(len(all_mpi_rgb_src)):
        mpi_rgb_src = all_mpi_rgb_src[i]
        mpi_disparity_src = all_mpi_disparity_src[i]
        mpi_sigma_src = all_mpi_sigma_src[i]
        xyz_tgt_BS3HW = all_xyz_tgt_BS3HW[i]
        G_tgt_src = all_G_tgt_src[i]
        K_src_inv = all_K_src_inv[i]
        B, S, _, H, W = mpi_rgb_src.size()

        mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

        # note that here we concat the mpi_src with xyz_tgt, because H_sampler will sample them for tgt frame
        # mpi_src is the same in whatever frame, but xyz has to be in tgt frame
        mpi_xyz_src = torch.cat((mpi_rgb_src, mpi_sigma_src, xyz_tgt_BS3HW), dim=2)  # BxSx(3+1+3)xHxW

        # homography warping of mpi_src into tgt frame
        G_tgt_src_Bs44 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 4, 4)  # Bsx4x4
        K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
        K_tgt_Bs33 = K_tgt.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3

        # BsxCxHxW, BsxHxW
        tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = H_sampler.sample(mpi_xyz_src.view(B*S, 7, H, W),
                                                            mpi_depth_src.view(B*S),
                                                            G_tgt_src_Bs44,
                                                            K_src_inv_Bs33,
                                                            K_tgt_Bs33)

        # mpi composition
        tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 7, H, W)
        tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
        tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
                                torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device),
                                torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device))
        tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW
        if (i==0):
            tgt_mpi_xyz_mv =torch.unsqueeze(tgt_mpi_xyz,dim=1)
            tgt_mask_mv = tgt_mask
        else:
            tgt_mpi_xyz_mv = torch.cat([tgt_mpi_xyz_mv,torch.unsqueeze(tgt_mpi_xyz,dim=1)],dim=1)
            tgt_mask_mv = torch.logical_and(tgt_mask_mv,tgt_mask)

    
    tgt_rgb_BS3HW = tgt_mpi_xyz_mv[:,:, :, 0:3, :, :].mean(dim=1,keepdim=False)
    tgt_sigma_BS1HW = tgt_mpi_xyz_mv[:,:, :, 3:4, :, :].mean(dim=1,keepdim=False)
    # To DO Change this to another better implementation
    tgt_xyz_BS3HW = tgt_mpi_xyz_mv[:,:, :, 4:, :, :].mean(dim=1,keepdim=False)

    # tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
    # tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
    #                             torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device),
    #                             torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device))

    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))
    tgt_rgb_syn, tgt_depth_syn, _, _ = render(tgt_rgb_BS3HW, tgt_sigma_BS1HW, tgt_xyz_BS3HW,
                                              use_alpha=use_alpha,
                                              is_bg_depth_inf=is_bg_depth_inf)
    # tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW

    return tgt_rgb_syn, tgt_depth_syn, tgt_mask_mv

def render_tgt_rgb_depth(src_view_norm_para_BS3,
                         H_sampler: HomographySample,
                         mpi_rgb_src,
                         mpi_sigma_src,
                         mpi_disparity_src,
                         xyz_tgt_BS3HW,
                         G_tgt_src,
                         K_src_inv, K_tgt,
                         use_alpha=False,
                         is_bg_depth_inf=False,
                         src_addr=None,
                         tgt_addr=None):
    """
    :param H_sampler:
    :param mpi_rgb_src: BxSx3xHxW
    :param mpi_sigma_src: BxSx1xHxW
    :param mpi_disparity_src: BxS
    :param xyz_tgt_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :param K_src_inv: Bx3x3
    :param K_tgt: Bx3x3
    :return:
    """
    B, S, _, H, W = mpi_rgb_src.size()
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    # note that here we concat the mpi_src with xyz_tgt, because H_sampler will sample them for tgt frame
    # mpi_src is the same in whatever frame, but xyz has to be in tgt frame
    mpi_xyz_src = torch.cat((mpi_rgb_src, mpi_sigma_src.to(mpi_rgb_src), xyz_tgt_BS3HW.to(mpi_rgb_src)), dim=2)  # BxSx(3+1+3)xHxW

    # homography warping of mpi_src into tgt frame
    G_tgt_src_Bs44 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 4, 4)  # Bsx4x4
    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
    K_tgt_Bs33 = K_tgt.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3

    # BsxCxHxW, BsxHxW
    tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = H_sampler.sample(src_view_norm_para_BS3,mpi_xyz_src.view(B*S, 7, H, W),
                                                        mpi_depth_src.view(B*S),
                                                        G_tgt_src_Bs44,
                                                        K_src_inv_Bs33,
                                                        K_tgt_Bs33,
                                                        src_addr,
                                                        tgt_addr)

    # mpi composition
    tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 7, H, W)
    tgt_rgb_BS3HW = tgt_mpi_xyz[:, :, 0:3, :, :]
    tgt_sigma_BS1HW = tgt_mpi_xyz[:, :, 3:4, :, :]
    tgt_xyz_BS3HW = tgt_mpi_xyz[:, :, 4:, :, :]

    tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
    tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
                                torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device),
                                torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device))

    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))

    # zmf
    sub_ims_tgt_torch_BS4HW = torch.cat([tgt_rgb_BS3HW,tgt_sigma_BS1HW],dim=2)
    tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW

    # tgt_rgb_syn, tgt_depth_syn, _, _ = render(tgt_rgb_BS3HW, tgt_sigma_BS1HW, tgt_xyz_BS3HW, use_alpha=True)
    # return tgt_rgb_syn, tgt_depth_syn, tgt_mask, sub_ims_tgt_torch_BS4HW
    return tgt_mask, sub_ims_tgt_torch_BS4HW, tgt_xyz_BS3HW

def predict_mpi_coarse_to_fine(mpi_predictor, src_imgs, xyz_src_BS3HW_coarse,
                               disparity_coarse_src, S_fine, is_bg_depth_inf):
    if S_fine > 0:
        with torch.no_grad():
            # predict coarse mpi
            mpi_coarse_src_list = mpi_predictor(src_imgs, disparity_coarse_src)  # BxS_coarsex4xHxW
            mpi_coarse_rgb_src = mpi_coarse_src_list[0][:, :, 0:3, :, :]  # BxSx1xHxW
            mpi_coarse_sigma_src = mpi_coarse_src_list[0][:, :, 3:, :, :]  # BxSx1xHxW
            _, _, _, weights = plane_volume_rendering(
                mpi_coarse_rgb_src,
                mpi_coarse_sigma_src,
                xyz_src_BS3HW_coarse,
                is_bg_depth_inf
            )
            weights = weights.mean((2, 3, 4)).unsqueeze(1).unsqueeze(2)

            # sample fine disparity
            disparity_fine_src = sample_pdf(disparity_coarse_src.unsqueeze(1).unsqueeze(2), weights, S_fine)
            disparity_fine_src = disparity_fine_src.squeeze(2).squeeze(1)

            # assemble coarse and fine disparity
            disparity_all_src = torch.cat((disparity_coarse_src, disparity_fine_src), dim=1) # Bx(S_coarse + S_fine)
            disparity_all_src, _ = torch.sort(disparity_all_src, dim=1, descending=True)
        mpi_all_src_list = mpi_predictor(src_imgs, disparity_all_src)  # BxS_coarsex4xHxW
        return mpi_all_src_list, disparity_all_src
    else:
        mpi_coarse_src_list = mpi_predictor(src_imgs, disparity_coarse_src)  # BxS_coarsex4xHxW
        return mpi_coarse_src_list, disparity_coarse_src

# pred_src_plane_alpha_decrease_factor = 20. # 5. # larger -> harder to change default alpha (mask)
# choose_tanh = True # True -> sharp edge in plane area
def prepare_plane_RGBA(pred_src_plane_RGBA_P4HW, pred_src_plane_masks_Phw, H, W, choose_tanh=True, pred_src_plane_alpha_decrease_factor=40.):
    # with torch.no_grad():
    pred_src_alpha_init_PHW_nograd = F.interpolate(
        pred_src_plane_masks_Phw[:,None,...],
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[:,0]
    pred_src_alpha_init_PHW_nograd = torch.clamp(input=pred_src_alpha_init_PHW_nograd,min=0.)

    # pred_src_plane_RGBA_P4HW = nn.Upsample(size=(H,W))(pred_src_plane_RGBA_P4HW)

    # print('zmf',torch.min(pred_src_plane_RGBA_P4HW[:,3,:,:]),torch.mean(pred_src_plane_RGBA_P4HW[:,3,:,:]),torch.max(pred_src_plane_RGBA_P4HW[:,3,:,:]))
    pred_src_plane_alpha_PHW = pred_src_plane_RGBA_P4HW[:,3,:,:] / pred_src_plane_alpha_decrease_factor

    alpha_add_PHW = pred_src_alpha_init_PHW_nograd + pred_src_plane_alpha_PHW
    tanh_alpha_PHW = alpha_add_PHW.tanh() # not sigmoid because in sigmoid, x > 0 -> y > 0.5, we don't like alpha > 0.5
    max_alpha_PHW = alpha_add_PHW / pred_src_alpha_init_PHW_nograd[pred_src_alpha_init_PHW_nograd>0.1].mean()

    # print('zmf',pred_src_alpha_init_PHW_nograd.requires_grad,tanh_alpha_PHW.requires_grad)

    pred_src_plane_RGBA_P4HW[:, 3,:,:] = torch.clamp(tanh_alpha_PHW,min=0.,max=1.) if choose_tanh else torch.clamp(max_alpha_PHW,min=0.,max=1.)
    pred_src_plane_RGBA_P4HW[:,:3,:,:] = pred_src_plane_RGBA_P4HW[:,:3,:,:].sigmoid()
    return pred_src_plane_RGBA_P4HW

def prepare_nonplane_A(pred_src_nonplane_A_1S1HW, pred_src_plane_masks_Phw, pred_src_plane_alpha_decrease_factor=40.):
    _, S, _, H, W = pred_src_nonplane_A_1S1HW.shape
    # with torch.no_grad():
    alpha_from_mask = F.interpolate(
        pred_src_plane_masks_Phw[:,None,...],
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[:,0]

    alpha_from_mask = alpha_from_mask.sigmoid()
    alpha_from_mask = torch.where(alpha_from_mask>0.5,alpha_from_mask,
                                  torch.zeros_like(alpha_from_mask,device=alpha_from_mask.device))
    plane_area_HW  = torch.sum(alpha_from_mask, dim=0)
    plane_area_HW = torch.clamp(plane_area_HW,min=0.,max=1.)
    nonplane_area_HW = 1-plane_area_HW

    ksize = H/30 if H/30 > 2 else 2
    ksize = int(ksize) if int(ksize) % 2 == 1 else int(ksize)+1
    max_pool = torch.nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize-1)/2))

    # ge0_PHW = torch.clamp(input=alpha_from_mask,min=0.)
    # ge0_HW  = torch.sum(ge0_PHW, dim=0).tanh()
    # de_HW  = ge0_HW * pred_src_plane_alpha_decrease_factor + 1.
    # pred_src_nonplane_A_1S1HW = pred_src_nonplane_A_1S1HW / de_HW[None,None,None,:,:].repeat(1,S,1,1,1)

    nonplane_mask_1S1HW = nonplane_area_HW[None, None, None, ...].repeat(1,S,1,1,1)
    tensor_erode_S1HW = -max_pool(-nonplane_mask_1S1HW[0])
    tensor_erode_dilate_S1HW = max_pool(tensor_erode_S1HW)
    pred_src_nonplane_A_1S1HW = pred_src_nonplane_A_1S1HW * tensor_erode_dilate_S1HW[None, ...]
    

    show = False
    if show:
        orig_mask = nonplane_mask_1S1HW[0,0,0] * 255.
        see_orig_mask = orig_mask.type(torch.uint8).detach().cpu().numpy()
        cv2.imwrite('./see_orig_mask.png',see_orig_mask)

        orig_mask = tensor_erode_dilate_S1HW[0,0] * 255.
        see_orig_mask = orig_mask.type(torch.uint8).detach().cpu().numpy()
        cv2.imwrite('./see_ed_mask.png',see_orig_mask)

        orig_mask = pred_src_nonplane_A_1S1HW[0,0,0] * 255.
        see_orig_mask = orig_mask.type(torch.uint8).detach().cpu().numpy()
        cv2.imwrite('./see_final_alpha.png',see_orig_mask)

        exit()

    return pred_src_nonplane_A_1S1HW

def plane_para4_trans( src_plane_para_BQ4, G_src_tgt_B44):
    # G_src_tgt = gt_src_pose @ torch.linalg.inv(gt_tgt_pose)
    B,Q,_ = src_plane_para_BQ4.shape

    G_src_tgt_transpose_B44 = G_src_tgt_B44.transpose(1,2)
    G_src_tgt_transpose_BQ44 = G_src_tgt_transpose_B44[:,None,...].repeat(1,Q,1,1)
    src_plane_para_BQ41 = src_plane_para_BQ4[...,None]

    src_plane_para_BQ4[:,:,3] = -src_plane_para_BQ4[:,:,3]
    tgt_view_plane_para_BQ4 = torch.matmul(G_src_tgt_transpose_BQ44, src_plane_para_BQ41)[:,:,:,0].float()
    tgt_view_plane_para_BQ4[:,:,3] = -tgt_view_plane_para_BQ4[:,:,3]
    src_plane_para_BQ4[:,:,3] = -src_plane_para_BQ4[:,:,3]
    return tgt_view_plane_para_BQ4

def plane_para_3to4(plane_para_3):
    if len(plane_para_3.shape) == 2:
        offset = torch.reciprocal(torch.norm(plane_para_3, dim=1, keepdim=True))
        plane_para_3 = plane_para_3 * offset
        plane_para_4 = torch.cat([plane_para_3,offset],dim=1)
    elif len(plane_para_3.shape) == 3:
        offset = torch.reciprocal(torch.norm(plane_para_3, dim=2, keepdim=True))
        plane_para_3 = plane_para_3 * offset
        plane_para_4 = torch.cat([plane_para_3,offset],dim=2)

    sign_correct = False
    if sign_correct:
        # the third in plane_para_3 has to be +
        # the third and the fourth in plane_para_4 has to be the same sign
        # two plane_para_4 with only sign difference, will be the same after converting to plane_para_3

        # below is not correct
        if len(plane_para_3.shape) == 2:
            plane_para_4abs = torch.abs(plane_para_4)
            sign = plane_para_4/plane_para_4abs
            sign = sign[:,1:2]
            plane_para_4 = plane_para_4 * sign.repeat(1,4)
        elif len(plane_para_3.shape) == 3:
            plane_para_4abs = torch.abs(plane_para_4)
            sign = plane_para_4/plane_para_4abs
            sign = sign[:,:,1:2]
            plane_para_4 = plane_para_4 * sign.repeat(1,1,4)
    return plane_para_4

def plane_para_4to3(plane_para_4):
    if len(plane_para_4.shape) == 2:
        plane_para_3 = plane_para_4[:,:3] / plane_para_4[:,3:]
    elif len(plane_para_4.shape) == 3:
        plane_para_3 = plane_para_4[:,:,:3] / plane_para_4[:,:,3:]
    return plane_para_3

def render_novel_view(mpi_plane_rgb_src_1P3HW,    mpi_plane_alpha_src_1P1HW,     plane_para_1P3, 
                        mpi_nonplane_rgb_src_1S3HW, mpi_nonplane_alpha_src_1S1HW,  nonplane_para_1S3,
                        G_tgt_src, K_src_inv, K_tgt, device_type, K_inv_dot_xy_1):
    B, P, _, H, W = mpi_plane_rgb_src_1P3HW.shape
    if os.environ['OLD_MPI'] == 'True': _, S, _, _, _ = mpi_nonplane_rgb_src_1S3HW.shape
    else: S = 0
    Q = P+S

    if os.environ['OLD_MPI'] == 'True':
        mpi_all_rgb_src_1Q3HW   = torch.cat([mpi_plane_rgb_src_1P3HW,   mpi_nonplane_rgb_src_1S3HW], dim=1)
        mpi_all_alpha_src_1Q1HW = torch.cat([mpi_plane_alpha_src_1P1HW, mpi_nonplane_alpha_src_1S1HW], dim=1)
        mpi_all_para_1Q3        = torch.cat([plane_para_1P3, nonplane_para_1S3], dim=1)
    else:
        mpi_all_rgb_src_1Q3HW = mpi_plane_rgb_src_1P3HW
        mpi_all_alpha_src_1Q1HW = mpi_plane_alpha_src_1P1HW
        mpi_all_para_1Q3 = plane_para_1P3


    if True:

        # src view plane para norm
        src_view_disparity_para_1Q1 = torch.norm(mpi_all_para_1Q3, dim=2, keepdim=True)
        src_view_norm_para_1Q3 = mpi_all_para_1Q3 * torch.reciprocal(src_view_disparity_para_1Q1)
        src_view_disparity_para_1Q = src_view_disparity_para_1Q1[:,:,0]

        # tgt view plane para
        mpi_all_para_1Q4 = plane_para_3to4(mpi_all_para_1Q3)
        para_tgt_1Q4 = plane_para4_trans(mpi_all_para_1Q4,torch.inverse(G_tgt_src))
        para_tgt_1Q3 = plane_para_4to3(para_tgt_1Q4)


        # src view 3D
        K_640 = torch.tensor([[577.,   0., 320.], [  0., 577., 240.], [  0.,   0.,   1.]])[None,...].repeat(B,1,1)
        K_640_inv = torch.inverse(K_640)
        mesh2 = HomographySample(H, W, device=device_type.device).meshgrid2

        xyz_src_1Q3HW = get_src_xyz_from_plane_para(
            mesh2,
            mpi_all_para_1Q3, # disparity_all_src,
            K_640_inv.to(device_type)
        )


        # tgt view 3D
        xyz_tgt_1Q3HW = get_tgt_xyz_from_plane_disparity(
            xyz_src_1Q3HW,
            G_tgt_src
        )


    tgt_mask_syn, sub_ims_tgt, tgt_xyz_BS3HW = render_tgt_rgb_depth(
        src_view_norm_para_1Q3,
        HomographySample(H, W, device=device_type.device), # self.homography_sampler_list[scale],
        mpi_all_rgb_src_1Q3HW.to(device_type),
        mpi_all_alpha_src_1Q1HW.to(device_type),
        src_view_disparity_para_1Q.to(device_type), # disparity_all_src,
        xyz_tgt_1Q3HW,
        G_tgt_src,
        K_src_inv,
        K_tgt,
        use_alpha=True,
        is_bg_depth_inf=False
    )


    # src RGB & depth
    sub_ims_src_1Q4HW = torch.cat([mpi_all_rgb_src_1Q3HW,mpi_all_alpha_src_1Q1HW],dim=2)
    para_src_1Q3 = mpi_all_para_1Q3
    pred_src_RGB_13HW, pred_src_depth_11HW, pred_src_alpha_acc_11HW = zmf_render(sub_ims_src_1Q4HW, para_src_1Q3, K_inv_dot_xy_1, xyz_src_1Q3HW)

    # target RGB & depth
    sub_ims_tgt_1Q4HW = sub_ims_tgt
    large = False
    if large:
        # print(sub_ims_tgt_1Q4HW.shape,K_inv_dot_xy_1.shape,tgt_xyz_BS3HW.shape)
        sub_ims_tgt_1Q4HW = nn.Upsample(size=(468,624))(sub_ims_tgt_1Q4HW[0])[None,...]
        tgt_xyz_BS3HW = nn.Upsample(size=(468,624))(tgt_xyz_BS3HW[0])[None,...]
        zeros = torch.zeros([3,468*624]).to(K_inv_dot_xy_1)
        zeros[:,:98304] = K_inv_dot_xy_1
        zeros[:,98304:98304*2] = K_inv_dot_xy_1
        K_inv_dot_xy_1 = zeros

        mpinum = 1
        sub_ims_tgt_1Q4HW = sub_ims_tgt_1Q4HW.repeat(1,mpinum,1,1,1).cpu()
        para_tgt_1Q3 = para_tgt_1Q3.repeat(1,mpinum,1).cpu()
        tgt_xyz_BS3HW = tgt_xyz_BS3HW.repeat(1,mpinum,1,1,1).cpu()
        K_inv_dot_xy_1 = K_inv_dot_xy_1.cpu()
        Q = Q*mpinum
        print(sub_ims_tgt_1Q4HW.shape,K_inv_dot_xy_1.shape,tgt_xyz_BS3HW.shape)

        
    time_start = time.perf_counter()
    pred_tgt_RGB_13HW, pred_tgt_depth_11HW, pred_tgt_alpha_acc_11HW = zmf_render(sub_ims_tgt_1Q4HW, para_tgt_1Q3, K_inv_dot_xy_1, tgt_xyz_BS3HW)
    # pred_tgt_RGB_13HW, pred_tgt_depth_11HW = zmf_render(sub_ims_tgt_1Q4HW, para_tgt_1Q3, K_inv_dot_xy_1, xyz_tgt_1Q3HW)
    time_end = time.perf_counter()


    test_mpi = False
    if test_mpi:
        time_start = time.perf_counter()
        pred_tgt_RGB_13HW, pred_tgt_depth_11HW, pred_tgt_alpha_acc_11HW, _ = render(sub_ims_tgt_1Q4HW[:,:,:3,:,:],sub_ims_tgt_1Q4HW[:,:,3:,:,:],tgt_xyz_BS3HW,use_alpha=True)
        # pred_tgt_RGB_13HW, pred_tgt_depth_11HW = zmf_render(sub_ims_tgt_1Q4HW, para_tgt_1Q3, K_inv_dot_xy_1, xyz_tgt_1Q3HW)
        time_end = time.perf_counter()
        print(time_end-time_start,'ss',Q)

    return {
        "tgt_mask_syn": tgt_mask_syn, # 1 1 H W
        "sub_ims_tgt": sub_ims_tgt, # 1 P 4 H W

        "pred_src_RGB_3HW": pred_src_RGB_13HW[0],
        "pred_tgt_RGB_3HW": pred_tgt_RGB_13HW[0],
        "pred_src_alpha_acc_1HW": pred_src_alpha_acc_11HW[0],
        "pred_tgt_alpha_acc_1HW": pred_tgt_alpha_acc_11HW[0],

        "pred_src_depth_HW": pred_src_depth_11HW[0][0],
        "pred_tgt_depth_HW": pred_tgt_depth_11HW[0][0],
    }

def zmf_render(sub_ims_BS4HW, plane_para_BS3, k_inv_dot_xy1, xyz_BQ3HW):

    B,S,_,h,w = sub_ims_BS4HW.shape
    rgb_BS3HW = []
    sigma_BS1HW = []
    depth_BS1HW = []

    for batchi in range(B):
        depth_maps_inv = torch.matmul(plane_para_BS3[batchi], k_inv_dot_xy1.to(plane_para_BS3))
        depth_maps_inv = torch.clamp(depth_maps_inv, min=0.01, max=1e4)
        depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)

        
        # SEE DEPTH MAPS
        # for i in range(depth_maps.shape[0]):
        #     dm = depth_maps[i].cpu().numpy().reshape(h, w)
        #     dm = np.clip(dm, a_min=1e-4, a_max=10.)
        #     depth_color = drawDepthImage(dm)
        #     depth_mask = dm > 1e-4
        #     depth_mask = depth_mask[:, :, np.newaxis]
        #     depth_color = depth_color * depth_mask
        #     depth_path = '%s/dm_0%d_tgt.png'%(save_path,i)
        #     cv2.imwrite(depth_path, depth_color)

        sub_ims = sub_ims_BS4HW.permute(0,3,4,1,2)[batchi] # h, w, S, 4
        sub_ims = torch.reshape(sub_ims,(h*w,S,4)).to(plane_para_BS3)

        sub_depths = xyz_BQ3HW.permute(0,3,4,1,2)[batchi] # h, w, S, 3
        sub_depths = torch.reshape(sub_depths,(h*w,S,3)).to(plane_para_BS3)

        sub_ims_R = sub_ims[:,:,0]
        sub_ims_G = sub_ims[:,:,1]
        sub_ims_B = sub_ims[:,:,2]
        sub_ims_A = sub_ims[:,:,3]

        sub_depth_D = sub_depths[:,:,2]

        

        depth_order = torch.argsort(depth_maps.t(), dim=1)
        sub_ims_R = torch.gather(sub_ims_R, 1, depth_order)
        sub_ims_G = torch.gather(sub_ims_G, 1, depth_order)
        sub_ims_B = torch.gather(sub_ims_B, 1, depth_order)
        sub_ims_A = torch.gather(sub_ims_A, 1, depth_order)
        sub_depth_D = torch.gather(sub_depth_D, 1, depth_order)


        sub_RGBs = torch.stack([sub_ims_R,sub_ims_G,sub_ims_B])
        sub_RGBs = sub_RGBs.permute(2,0,1)
        rgb_BS3HW.append(torch.reshape(sub_RGBs,(S,3,h,w)))
        
        sub_ims_A = sub_ims_A[None, ...]
        sub_ims_A = sub_ims_A.permute(2,0,1)
        sigma_BS1HW.append(torch.reshape(sub_ims_A,(S,1,h,w)))

        sub_depth_D = sub_depth_D[None, ...]
        sub_depth_D = sub_depth_D.permute(2,0,1)
        depth_BS1HW.append(torch.reshape(sub_depth_D,(S,1,h,w)))

    rgb_BS3HW = torch.stack(rgb_BS3HW)
    sigma_BS1HW = torch.stack(sigma_BS1HW)
    depth_BS1HW = torch.stack(depth_BS1HW)

    imgs_syn, alpha_acc = alpha_composition(sigma_BS1HW, rgb_BS3HW)
    depth_syn, _ = alpha_composition(sigma_BS1HW, depth_BS1HW)

    return imgs_syn, depth_syn, alpha_acc

def render_everything(pred_plane_mask_orig_Phw, pred_src_plane_RGBA_P4HW, pred_src_plane_para_P3, pred_src_nonplane_RGBA_S4HW, gt_G_src_tgt_44, 
                        gt_src_view_rgb_3HW, K_inv_dot_xy_1, device_type, save_images=False, scale=0, use_src_view_rgb=False, pred_class_P=None):
    
    trick = True
    pred_src_plane_alpha_decrease_factor = 100.
    vis_results = {}
    results = {}

    P,_, H, W = pred_src_plane_RGBA_P4HW.shape
    if os.environ['SEG_NONPLANE'] == 'True':
        assert pred_src_nonplane_RGBA_S4HW == None
    else:
        S, _, H, W = pred_src_nonplane_RGBA_S4HW.shape

    
    if trick: # dilate plane mask to limit the plane-region alpha
        
        # old trick 
        # pred_src_plane_RGBA_P4HW = prepare_plane_RGBA(pred_src_plane_RGBA_P4HW, pred_plane_mask_orig_Phw, H, W, pred_src_plane_alpha_decrease_factor)
        erode_kernel = H/float(os.environ['ERODE_FAC'])
        if erode_kernel < 2: erode_kernel =  2
        erode_kernel = int(erode_kernel) if int(erode_kernel) % 2 == 1 else int(erode_kernel)+1
        max_pool = torch.nn.MaxPool2d(kernel_size=erode_kernel, stride=1, padding=int((erode_kernel-1)/2))

        dilated_mask_o = F.interpolate(pred_plane_mask_orig_Phw[:,None,:,:], size=pred_src_plane_RGBA_P4HW.shape[-2:], 
                                    mode="bilinear", align_corners=False).sigmoid()

        full_mask = True
        if not full_mask:
            dilated_mask = torch.where(dilated_mask_o>0.5,dilated_mask_o,torch.zeros_like(dilated_mask_o,device=dilated_mask_o.device))
            dilated_mask = max_pool(dilated_mask) > 0.5
        else:
            # cur_prob_masks = pred_class_P.view(-1, 1, 1, 1) * dilated_mask_o # P, 1, H, W
            # cur_mask_ids = cur_prob_masks[:,0,:,:].argmax(0) # h, w
            # dilated_mask = torch.zeros(dilated_mask_o.shape,device=dilated_mask_o.device)
            # for pi in range(P):
            #     dilated_mask[pi,0] = (cur_mask_ids == pi).float()
            # dilated_mask = max_pool(dilated_mask) > 0.5

            cur_mask_ids = dilated_mask_o[:,0,:,:].argmax(0) # h, w
            dilated_mask = torch.zeros(dilated_mask_o.shape,device=dilated_mask_o.device)
            for pi in range(P):
                dilated_mask[pi,0] = (cur_mask_ids == pi).float()
            dilated_mask = max_pool(dilated_mask) > 0.5


        filtered_alpha = pred_src_plane_RGBA_P4HW[:,3:,:,:] * dilated_mask.to(pred_src_plane_RGBA_P4HW)
        pred_src_plane_RGBA_P4HW = torch.cat([pred_src_plane_RGBA_P4HW[:,:3,...], filtered_alpha.to(pred_src_plane_RGBA_P4HW)], dim=1)


    if use_src_view_rgb:
        pred_src_plane_RGBA_P4HW[:,:3,:,:] = gt_src_view_rgb_3HW[None,...].repeat(P,1,1,1)
        if os.environ['OLD_MPI'] == 'True':
            pred_src_nonplane_RGBA_S4HW[:,:3,:,:] = gt_src_view_rgb_3HW[None,...].repeat(S,1,1,1)
    
    if save_images:
        vis_results['pred_src_plane_RGBA_P4HW'] = pred_src_plane_RGBA_P4HW

    scale_fac = W/640.
    K = torch.tensor([ [577.*scale_fac, 0., W/2.], [0., 577.*scale_fac, H/2.], [0.,0.,1.], ]).to(device_type)[None,...]
    G_src_tgt_144 = gt_G_src_tgt_44[None,...]
    K_inv = torch.inverse(K.float()).to(device_type)
    G_tgt_src_144 = torch.inverse(G_src_tgt_144.float()).to(device_type)

    pred_src_plane_RGBA_1P4HW = pred_src_plane_RGBA_P4HW[None,...]
    pred_src_plane_RGB_1P3HW  = pred_src_plane_RGBA_1P4HW[:,:,:3,:,:]
    pred_src_plane_A_1P1HW    = pred_src_plane_RGBA_1P4HW[:,:,3:,:,:]

    if os.environ['OLD_MPI'] == 'True':
        # prepare non-plane parameters
        start_disparity = 0.999
        end_disparity = 0.001
        src_nonplane_disparity_S = torch.linspace(
            start_disparity, end_disparity, S, dtype=device_type.dtype, device=device_type.device
        )
        src_nonplane_depth_S3 = torch.reciprocal(src_nonplane_disparity_S)[:,None].repeat(1,3)
        # src_nonplane_depth_S3 = src_nonplane_disparity_S[:,None].repeat(1,3)
        n_3 = torch.tensor([0, 0, 1], dtype=device_type.dtype, device=device_type.device)
        n_S3 = n_3[None,:].repeat(S,1)
        src_nonplane_para_S3 = n_S3 / src_nonplane_depth_S3
        src_nonplane_para_1S3 = src_nonplane_para_S3[None,...]

        # prepare non-plane rgba
        pred_src_nonplane_RGBA_1S4HW = pred_src_nonplane_RGBA_S4HW[None,...]
        pred_src_nonplane_RGB_1S3HW  = pred_src_nonplane_RGBA_1S4HW[:,:,:3,:,:]
        pred_src_nonplane_A_1S1HW    = pred_src_nonplane_RGBA_1S4HW[:,:,3:,:,:]

        pred_src_nonplane_A_1S1HW = prepare_nonplane_A(pred_src_nonplane_A_1S1HW, pred_plane_mask_orig_Phw, pred_src_plane_alpha_decrease_factor)

        if save_images:
            vis_results['pred_src_nonplane_RGB_1S4HW'] = torch.cat([pred_src_nonplane_RGB_1S3HW,pred_src_nonplane_A_1S1HW],dim=2)
    else:
        pred_src_nonplane_RGB_1S3HW = None
        pred_src_nonplane_A_1S1HW = None
        src_nonplane_para_1S3 = None


    result_dic = render_novel_view(pred_src_plane_RGB_1P3HW,    pred_src_plane_A_1P1HW,    pred_src_plane_para_P3[None,...],
                                   pred_src_nonplane_RGB_1S3HW, pred_src_nonplane_A_1S1HW, src_nonplane_para_1S3,
                                   G_tgt_src=G_tgt_src_144, K_src_inv=K_inv, K_tgt=K, device_type=device_type,
                                   K_inv_dot_xy_1=K_inv_dot_xy_1[scale])
    
    # print(result_dic.keys()) # dict_keys(['tgt_mask_syn', 'sub_ims_tgt', 'pred_src_RGB_3HW', 'pred_tgt_RGB_3HW', 'pred_src_depth_HW', 'pred_tgt_depth_HW'])

    # target mask
    pred_tgt_vmask_HW = result_dic["tgt_mask_syn"][0,0]
    pred_tgt_vmask_HW = torch.ge(pred_tgt_vmask_HW, torch.mean(pred_tgt_vmask_HW))
    results['pred_tgt_vmask_HW'] = pred_tgt_vmask_HW

    results['pred_src_RGB_3HW'] = result_dic["pred_src_RGB_3HW"]
    results['pred_tgt_RGB_3HW'] = result_dic["pred_tgt_RGB_3HW"]
    results['pred_src_depth_HW'] = result_dic["pred_src_depth_HW"]
    results['pred_tgt_depth_HW'] = result_dic["pred_tgt_depth_HW"]
    results['pred_src_alpha_acc_1HW'] = result_dic["pred_src_alpha_acc_1HW"]
    results['pred_tgt_alpha_acc_1HW'] = result_dic["pred_tgt_alpha_acc_1HW"]

    return results, vis_results