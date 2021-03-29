import os

from cruw.cruw import CRUW
from cruw.parser.config_parser import load_loc3d_cam_config_dict, load_human_anno_config_dict
from cruw.parser.read_det_files import read_dets_kitti_txt, read_dets_crf_txt, read_ra_labels_csv, \
    read_image_labels_csv


class CamResLoader:
    """ Loader of detection results from camera. """

    def __init__(self,
                 data_root: str,
                 dataset: CRUW,
                 config_name: str):
        self.data_root = data_root
        self.dataset = dataset
        self.config_name = config_name
        self.loc3d_cam_cfg = load_loc3d_cam_config_dict(self.config_name)
        self.seq_names = self.loc3d_cam_cfg.seq_names
        if self.loc3d_cam_cfg.gt_root is not None:
            self.gt_exist = True
        else:
            self.gt_exist = False

    def load(self, verbose=False):
        cam_res_dict = {}
        if verbose:
            print("Loading results ...")
        for seq_name in self.seq_names:
            if self.loc3d_cam_cfg.date_included:
                date = seq_name[:10]
                seq_res_dir = os.path.join(self.loc3d_cam_cfg.res_root, date, seq_name,
                                           self.loc3d_cam_cfg.res_dir_name)
            else:
                seq_res_dir = os.path.join(self.loc3d_cam_cfg.res_root, seq_name,
                                           self.loc3d_cam_cfg.res_dir_name)
            cam_res_dict[seq_name] = self._load_loc3d_cam_seq(seq_res_dir)
        if self.gt_exist:
            gt_dict = {}
            if verbose:
                print("Loading ground truth ...")
            for seq_name in self.seq_names:
                if self.loc3d_cam_cfg.date_included:
                    date = seq_name[:10]
                    seq_gt_dir = os.path.join(self.loc3d_cam_cfg.gt_root, date, seq_name,
                                              self.loc3d_cam_cfg.gt_dir_name)
                else:
                    seq_gt_dir = os.path.join(self.loc3d_cam_cfg.gt_root, seq_name,
                                              self.loc3d_cam_cfg.gt_dir_name)
                gt_dict[seq_name] = self._load_gt_seq(seq_gt_dir)
        else:
            gt_dict = None
        return cam_res_dict, gt_dict

    def _load_loc3d_cam_seq(self, seq_res_dir) -> list:
        """
        Load camera detection (3d localization) results for a sequence
        :param seq_res_dir: Full result path
        :return: A list of all detections in all frame
        """
        results = []
        if os.path.exists(seq_res_dir):
            assert self.loc3d_cam_cfg.res_format == 'KITTI' or self.loc3d_cam_cfg.res_format == 'CRF' or self.loc3d_cam_cfg.res_format == 'CSV', \
                'Object 3D localization format {} cannot be recognized.'.format(self.loc3d_cam_cfg.res_format)
            if self.loc3d_cam_cfg.res_format == 'CSV':
                results = read_image_labels_csv(seq_res_dir)
                return results
            label_names = sorted(os.listdir(seq_res_dir))
            for label_name in label_names:
                frame_id = int(label_name[:-4])
                txt_path = os.path.join(seq_res_dir, label_name)
                if self.loc3d_cam_cfg.res_format == 'KITTI':
                    obj_info = read_dets_kitti_txt(txt_path, self.dataset)
                elif self.loc3d_cam_cfg.res_format == 'CRF':
                    obj_info = read_dets_crf_txt(txt_path, self.dataset)
                else:
                    raise NotImplementedError
                results.append(obj_info)
        else:
            # TODO: other result formats
            print('Warning: file not found %s' % seq_res_dir)
            return None
        return results

    def _load_gt_seq(self, seq_res_dir) -> list:
        """
        Load camera detection (3d localization) results for a sequence
        :param seq_res_dir: Full result path
        :return: A list of all detections in all frame
        """
        if os.path.exists(seq_res_dir):
            if self.loc3d_cam_cfg.gt_format == "VIA_CSV":
                results = read_ra_labels_csv(seq_res_dir, self.dataset)
            else:
                raise NotImplementedError
        else:
            print('Warning: file not found %s' % seq_res_dir)
            return None
        return results


class HumanAnnoLoader:
    """ Loader of human annotations on radar data. """

    def __init__(self,
                 data_root: str,
                 dataset: CRUW,
                 config_name: str):
        self.data_root = data_root
        self.dataset = dataset
        self.config_name = config_name
        self.anno_cfg = load_human_anno_config_dict(self.config_name)
        self.seq_names = self.anno_cfg.seq_names

    def load(self, verbose=False):
        gt_dict = {}
        if verbose:
            print("Loading ground truth ...")
        for seq_name in self.seq_names:
            if self.anno_cfg.date_included:
                date = seq_name[:10]
                seq_gt_dir = os.path.join(self.anno_cfg.gt_root, date, seq_name,
                                          self.anno_cfg.gt_dir_name)
            else:
                seq_gt_dir = os.path.join(self.anno_cfg.gt_root, seq_name,
                                          self.anno_cfg.gt_dir_name)
            gt_dict[seq_name] = self._load_gt_seq(seq_gt_dir)
        return gt_dict

    def _load_gt_seq(self, seq_res_dir) -> list:
        """
        Load camera detection (3d localization) results for a sequence
        :param seq_res_dir: Full result path
        :return: A list of all detections in all frame
        """
        if os.path.exists(seq_res_dir):
            if self.anno_cfg.gt_format == "VIA_CSV":
                results = read_ra_labels_csv(seq_res_dir, self.dataset)
            else:
                raise NotImplementedError
        else:
            print('Warning: file not found %s' % seq_res_dir)
            return None
        return results
