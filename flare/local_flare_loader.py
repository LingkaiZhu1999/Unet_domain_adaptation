# File: local_flare_loader.py
# (Contents are identical to the previous answer. No changes needed.)

import os
import datasets

_DESCRIPTION = "Local loader for the FLARE 2024 Task 3 Domain Adaptation Dataset."

class FlareTask3Config(datasets.BuilderConfig):
    def __init__(self, image_dir, label_dir=None, has_labels=True, **kwargs):
        super(FlareTask3Config, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.has_labels = has_labels

class LocalFlareTask3(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        FlareTask3Config(
            name="train_ct_gt", description="Training CT scans with ground-truth labels (50 scans)",
            image_dir="train_CT_gt_label/imagesTr", label_dir="train_CT_gt_label/labelsTr",
        ),
        FlareTask3Config(
            name="train_ct_pseudo_blackbean_flare22", description="Training CT scans with pseudo labels from blackbean_flare22",
            image_dir="train_CT_pseudolabel/imagesTr", label_dir="train_CT_pseudolabel/pseudo_label_blackbean_flare22",
        ),
        FlareTask3Config(
            name="train_mri_unlabeled", description="Unlabeled training MRI scans (4817 scans)",
            image_dir="train_MRI_unlabeled", has_labels=False,
        ),
        FlareTask3Config(
            name="train_pet_unlabeled", description="Unlabeled training PET scans (1000 scans)",
            image_dir="train_PET_unlabeled", has_labels=False,
        ),
        FlareTask3Config(
            name="validation_mri", description="Validation MRI scans with labels (110 scans)",
            image_dir="validation/MRI_imagesVal", label_dir="validation/MRI_labelsVal",
        ),
        FlareTask3Config(
            name="validation_pet", description="Validation PET scans with labels (50 scans)",
            image_dir="validation/PET_imagesVal", label_dir="validation/PET_labelsVal",
        ),
        FlareTask3Config(
            name="coreset_train_unlabeled_mri", description="Test MRI scans (100 scans)",
            image_dir="MRI_unlabeled_100_random", has_labels=False,
        ),
        FlareTask3Config(
            name="coreset_train_unlabeled_pet", description="Test PET scans (100 scans)",
            image_dir="PET_unlabeled_100_random", has_labels=False,
        ),
    ]

    def _info(self):
        features = {"image_path": datasets.Value("string"), "label_path": datasets.Value("string")}
        return datasets.DatasetInfo(description=_DESCRIPTION, features=datasets.Features(features))

    def _split_generators(self, dl_manager):
        if not dl_manager.manual_dir:
            raise ValueError("You must specify `data_dir` in `load_dataset`.")
        data_root_path = dl_manager.manual_dir

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
            "image_dir": os.path.join(data_root_path, self.config.image_dir),
            "label_dir": os.path.join(data_root_path, self.config.label_dir) if self.config.has_labels else None,
            "has_labels": self.config.has_labels,
        })]

    def _generate_examples(self, image_dir, label_dir, has_labels):
        """This is the ROBUST recursive version. It's the best choice."""
        key = 0
        for root, dirs, files in os.walk(image_dir):
            for filename in sorted(files):
                if filename.endswith((".nii.gz", ".nii")):
                    image_path = os.path.join(root, filename)
                    if has_labels:
                        if filename.endswith("_0000.nii.gz"):
                            label_filename = filename.replace("_0000.nii.gz", ".nii.gz")
                        else:
                            label_filename = filename
                        label_path = os.path.join(label_dir, label_filename)
                        if os.path.exists(label_path):
                            yield key, {"image_path": image_path, "label_path": label_path}
                            key += 1
                    else:
                        yield key, {"image_path": image_path, "label_path": "N/A"}
                        key += 1

