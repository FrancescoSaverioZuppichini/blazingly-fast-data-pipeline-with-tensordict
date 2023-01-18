from data import get_image_and_labels, ObjectDetectionData
from pathlib import Path

# print(get_image_and_labels(
#     Path(
#         "/home/zuppif/Documents/medium/blazingly-fast-data-pipeline-with-tensordict/data/train/images/21_2_1646933460_jpg.rf.110733eeece88f6176eea9f01b732669.jpg"
#     )
# ))


ObjectDetectionData.from_dataset(Path("/home/zuppif/Documents/medium/blazingly-fast-data-pipeline-with-tensordict/data/train/images/"))