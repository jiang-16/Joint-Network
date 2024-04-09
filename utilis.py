import numpy as np
import odl
from scipy.io import loadmat
from fliter_proj import *
import matplotlib.pyplot as plt
class CTReconstruction:
    def __init__(self, projections, geo):
        self.projections = projections
        self.geo = geo

    def preprocess(self):
        # 设置几何参数
        num_detectors = self.projections.shape[0]       # 探测器单元数量
        num_angles = self.projections.shape[1]          # 角度数

        # 角度范围为 0 到 2pi
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)


        # 假设扇形束射线半径和探测器宽度，这些参数应该与实际扫描参数匹配
        detector_size = 0.3  # 探测器元素的大小（实际单元大小）
        detector_span = num_detectors * detector_size  # 探测器行的总宽度

        # 进行预处理
        self.geo["sDetector"] = self.geo["nDetector"] * self.geo["dDetector"]
        self.geo["sVoxel"] = self.geo["nVoxel"] * self.geo["dVoxel"]

        return angles, detector_span

    def setup_geometry(self, angles, detector_span):
        # 创建投影空间，可能需要类型根据你的数据进行调整
        reco_space = odl.uniform_discr([-25.6, -25.6], [25.6, 25.6], [512, 512], dtype='float32')
        angle_partition=odl.uniform_partition(0, 2 * np.pi, len(angles))
        detector_partition=odl.uniform_partition(-detector_span / 2, detector_span / 2,704)
        src_radius=500
        det_radius=1000

        # 创建扇形束几何
        geometry = odl.tomo.FanFlatGeometry(angle_partition,detector_partition,src_radius, det_radius)

        return geometry, reco_space

    def create_ray_transform(self, geometry, reco_space):
        # 创建前向算子
        ray_transform = odl.tomo.RayTransform(reco_space, geometry)
        return ray_transform

    def filtered_projection(self, angles):
        # Assume `proj` is predefined or loaded data.
        proj = np.flip(self.projections,axis=1)
        filtered_proj = filtered(proj, self.geo, angles)
        return filtered_proj

    def reconstruct(self):
        angles, detector_span = self.preprocess()
        geometry, reco_space = self.setup_geometry(angles, detector_span)

        ray_transform = self.create_ray_transform(geometry, reco_space)
        filtered_proj = self.filtered_projection(angles)
        fbp = ray_transform.adjoint(filtered_proj.T)

        fbp = fbp.data.T
        fbp = fbp.clip(min=0)

        return fbp
'''
data = loadmat('./Train/proj/proj_0016_mask.mat')
projections = data['proj2']  # 假设数据变量的名字叫 'name_of_the_variable_inside_mat'
geo = {
    "DSD": 1200,
    "DSO": 600,
    "nDetector": np.array([704, 1]),
    "dDetector": np.array([1, 1]) * 0.23,
    "sDetector": None,
    "nVoxel": np.array([512, 512]) / 1,
    "dVoxel": np.array([1, 1]) * 0.1,
    "sVoxel": None,
    "detoffset": np.array([0, 0]),
    "orgoffset": np.array([0, 0, 0])
}

reconstructor = CTReconstruction(projections, geo)
fbp = reconstructor.reconstruct()
plt.imshow(fbp,'gray')
plt.show()
'''


