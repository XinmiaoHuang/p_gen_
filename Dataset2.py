import torch
import os
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

"""
some code are borrowed from vunet and pytorch tutorial
https://github.com/CompVis/vunet.git
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""


class DeepfashionPoseDataset(Dataset):
    """Deepfashion dataset with pose. """
    def __init__(self, img_shape, base_dir, index_dir, map_dir, transform=None, fetch_target=True, training=True):
        self.base_dir = base_dir
        self.index_dir = index_dir
        self.map_dir = map_dir
        self.transform = transform
        self.fetch_target = fetch_target
        self.training = training
        with open(index_dir, 'rb') as f:
            self.index = pickle.load(f)
        self.train = []
        self.test = []
        self.train_joint = []
        self.test_joint = []
        for idx, path in enumerate(self.index['imgs']):
            if path.startswith('train'):
                self.train.append(path)
                self.train_joint.append(self.index["joints"][idx])
            else:
                self.test.append(path)
                self.test_joint.append(self.index["joints"][idx])
        self.joint_order = self.index['joint_order']
        self.img_shape = img_shape
        # scale the joints coordinate
        h, w = self.img_shape[:2]
        wh = np.array([[[w, h]]])
        self.train_joint = self.train_joint * wh
        self.test_joint = self.test_joint * wh

    def __len__(self):
        if self.training:
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        if self.training:
            current_set = self.train
            current_joint = self.train_joint
        else:
            current_set = self.test
            current_joint = self.test_joint
        img_dir = os.path.join(self.base_dir, current_set[idx])
        image = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_shape[:2])
        pose = self.make_joint_map(current_joint[idx])
        semantic_dir = os.path.join(self.map_dir, current_set[idx][:-4] + '.png')
        semantic_map = np.array(cv2.cvtColor(cv2.imread(semantic_dir), cv2.COLOR_BGR2RGB))
        semantic_map = cv2.resize(semantic_map, self.img_shape[:2])
        sample = {'image': image, 'pose': pose, 's_map': semantic_map}
        for key in sample.keys():
            if key != 'pose':
                sample[key] = sample[key] * 1./255
                sample[key] = sample[key].transpose((2, 0, 1))
            sample[key] = torch.from_numpy(sample[key])
            if self.transform:
                sample[key] = self.transform(sample[key])
        if self.fetch_target:
            input_dir = current_set[idx]
            target_set = [id for id,img in enumerate(current_set[idx-10:idx+10])if img.startswith(input_dir[:11])]
            if 10 in target_set:
                target_set.remove(10)
            if len(target_set) == 0:
                target_set.append(10)
            t_idx = random.choice(target_set) + idx - 10
            target_dir = os.path.join(self.base_dir, current_set[t_idx])
            target = cv2.cvtColor(cv2.imread(target_dir), cv2.COLOR_BGR2RGB)
            target = cv2.resize(target, self.img_shape[:2])
            t_pose = self.make_joint_map(current_joint[t_idx])
            t_semantic_dir = os.path.join(self.map_dir, current_set[t_idx][:-4] + '.png')
            t_semantic_map = np.array(cv2.cvtColor(cv2.imread(t_semantic_dir), cv2.COLOR_BGR2RGB))
            t_semantic_map = cv2.resize(t_semantic_map, self.img_shape[:2])
            target_sample = {'image': target, 'pose': t_pose, 't_map': t_semantic_map}
            for key in target_sample.keys():
                if key != 'pose':
                    target_sample[key] = target_sample[key] * 1. / 255
                    target_sample[key] = target_sample[key].transpose((2, 0, 1))
                if self.transform:
                    target_sample[key] = self.transform(target_sample[key])
                target_sample[key] = torch.from_numpy(target_sample[key])
            image_pair = {'input': sample, 'target': target_sample}
            return image_pair
        else:
            sample = {'input': sample}
            return sample

    def make_joint_map(self, joints, k_size=9):
        joint_map = np.zeros((18,) + self.img_shape[:-1], dtype=np.float)
        gaussian_vec = cv2.getGaussianKernel(k_size, 0)
        gaussian_matrix = np.dot(gaussian_vec, gaussian_vec.T)
        _range = np.max(gaussian_matrix) - np.min(gaussian_matrix)
        gaussian_matrix = (gaussian_matrix - np.min(gaussian_matrix)) / _range
        for idx, joint in enumerate(joints):
            y = int(joint[0])
            x = int(joint[1])
            if x >= 0 and y >= 0:
                init_x = x - k_size // 2
                init_y = y - k_size // 2
                for i in range(k_size):
                    for j in range(k_size):
                        new_x = init_x + i
                        new_y = init_y + j
                        if 0 <= new_x < self.img_shape[0] and 0 <= new_y < self.img_shape[1]:
                            joint_map[idx][new_x][new_y] += gaussian_matrix[i][j]
        return joint_map

    """
    https://github.com/CompVis/vunet.git
    """
    # def preprocess(self, x):
    #     """From uint8 image to [-1,1]."""
    #     return np.cast[np.float32](x / 127.5 - 1.0)

    @staticmethod
    def valid_joints(*joints):
        j = np.stack(joints)
        return (j >= 0).all()

    def get_crop(self, bpart, joints, jo, wh, o_w, o_h, ar=1.0):
        bpart_indices = [jo.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices])

        # fall backs
        if not self.valid_joints(part_src):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])

        if not self.valid_joints(part_src):
            return None

        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0], o_h - 1])
            part_src = np.float32([a, b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a, b, c, d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5 * (part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2 * neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha * normal
                b = part_src[0] - alpha * normal
                c = part_src[1] - alpha * normal
                d = part_src[1] + alpha * normal
                # part_src = np.float32([a,b,c,d])
                part_src = np.float32([b, c, d, a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha * normal
            b = part_src[0] - alpha * normal
            c = part_src[1] - alpha * normal
            d = part_src[1] + alpha * normal
            part_src = np.float32([a, b, c, d])

        dst = np.float32([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        return M

    def normalize(self, imgs, coords, stickmen, jo, box_factor):
        img = imgs
        joints = coords
        stickman = stickmen

        h, w = img.shape[:2]
        o_h = h
        o_w = w
        h = h // 2 ** box_factor
        w = w // 2 ** box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
            ["lshoulder", "lhip", "rhip", "rshoulder"],
            ["lshoulder", "rshoulder", "cnose"],
            ["lshoulder", "lelbow"],
            ["lelbow", "lwrist"],
            ["rshoulder", "relbow"],
            ["relbow", "rwrist"],
            ["lhip", "lknee"],
            ["rhip", "rknee"]]
        ar = 0.5

        part_imgs = list()
        part_stickmen = list()
        for bpart in bparts:
            part_img = np.zeros((h, w, 3))
            part_stickman = np.zeros((h, w, 3))
            M = self.get_crop(bpart, joints, jo, wh, o_w, o_h, ar)

            if M is not None:
                part_img = cv2.warpPerspective(img, M, (h, w), borderMode=cv2.BORDER_REPLICATE)
                part_stickman = cv2.warpPerspective(stickman, M, (h, w), borderMode=cv2.BORDER_REPLICATE)

            part_imgs.append(part_img)
            part_stickmen.append(part_stickman)
        img = np.concatenate(part_imgs, axis=2)
        stickman = np.concatenate(part_stickmen, axis=2)

        return img, stickman

    def make_joint_img(self, img_shape, jo, joints):
        # three channels: left, right, center
        scale_factor = img_shape[1] / 128
        thickness = int(3 * scale_factor)
        imgs = list()
        for i in range(3):
            imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part), :] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        right_lines = [
            ("rankle", "rknee"),
            ("rknee", "rhip"),
            ("rhip", "rshoulder"),
            ("rshoulder", "relbow"),
            ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color=255, thickness=thickness)

        left_lines = [
            ("lankle", "lknee"),
            ("lknee", "lhip"),
            ("lhip", "lshoulder"),
            ("lshoulder", "lelbow"),
            ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color=255, thickness=thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("cnose")]
        neck = 0.5 * (rs + ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        if np.min(a) >= 0 and np.min(b) >= 0:
            cv2.line(imgs[0], a, b, color=127, thickness=thickness)
            cv2.line(imgs[1], a, b, color=127, thickness=thickness)

        cn = tuple(np.int_(cn))
        leye = tuple(np.int_(joints[jo.index("leye")]))
        reye = tuple(np.int_(joints[jo.index("reye")]))
        if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
            cv2.line(imgs[0], cn, reye, color=255, thickness=thickness)
            cv2.line(imgs[1], cn, leye, color=255, thickness=thickness)

        img = np.stack(imgs, axis=-1)
        if img_shape[-1] == 1:
            img = np.mean(img, axis=-1)[:, :, None]
        return img

    def make_pose(self, training=True):
        base_target_img_dir = '../PoseGuide/VUNet/splited_dataset/imgs/'          # 这里改为分割数据集存放路径
        base_target_pose_dir = '../PoseGuide/VUNet/splited_dataset/poses/'        # 这里改为分割数据集pose存放路径
        import shutil
        current_set = self.train if training else self.test
        current_joint = self.train_joint if training else self.test_joint
        for i in range(len(current_set)):
            img_dir = os.path.join(self.base_dir, current_set[i])
            shutil.copyfile(img_dir, os.path.join(base_target_img_dir, current_set[i]))
            pose = self.make_joint_img(self.img_shape, self.joint_order, current_joint[i])
            current_set[i] = current_set[i][:-3] + 'png'
            cv2.imwrite(os.path.join(base_target_pose_dir, current_set[i]), pose)


def show_sample(imgs, poses, semantics, target):
    imgs = imgs.transpose((1,2,0))
    poses = np.sum(poses.transpose((1, 2, 0)), axis=2)
    semantics = semantics.transpose((1, 2, 0))
    target = target.transpose((1, 2, 0))
    plt.figure()
    plt.axis('off')
    plt.subplot(1, 4, 1)
    plt.imshow(imgs)
    plt.subplot(1, 4, 2)
    plt.imshow(poses)
    plt.subplot(1, 4, 3)
    plt.imshow(semantics)
    plt.subplot(1, 4, 4)
    plt.imshow(target)
    plt.show()


if __name__ == '__main__':
    base_dir = 'D:/Dataset/deepfashion'
    index_dir = 'D:/Dataset/deepfashion/index.p'
    map_dir = 'D:/Dataset/deepmap_test'
    dataset = DeepfashionPoseDataset((256, 256, 3), base_dir, index_dir, map_dir)
    # dataset.make_pose()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for index, sample in enumerate(dataloader):
        imgs = sample['input']['image'].numpy()
        poses = sample['target']['pose'].numpy()
        semantics = sample['target']['t_map'].numpy()
        target = sample['target']['image'].numpy()
        show_sample(imgs[0], poses[0], semantics[0], target[0])
    print("process ended.")
