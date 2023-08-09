import torch

class AnchorBox:
    def __init__(self):
        self._areas = [x ** 2 for x in [4.0, 7.0, 13.0, 26.0, 52.0]]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1/3, 2/3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims_all = []
        for area in self._areas: # 5
            anchor_dims = []
            for ratio in self.aspect_ratios: # 3
                anchor_height = torch.sqrt(torch.tensor(area/ratio))
                anchor_width  = area / anchor_height
                dims = torch.reshape(
                    torch.stack([anchor_width, anchor_height], dim=-1), [1, 1, 2]
                )

                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(torch.stack(anchor_dims, dim=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        rx = torch.arange(0, feature_width, dtype=torch.float32) + 0.5
        ry = torch.arange(0, feature_height, dtype=torch.float32) + 0.5

        centers = torch.stack(torch.meshgrid(rx, ry), dim=-1) * self._strides[level-3]
        centers = torch.unsqueeze(centers, dim=2)
        centers = torch.tile(centers, [1, 1, self._num_anchors, 1])

        dims = torch.tile(self._anchor_dims[level-3], [int(feature_height), int(feature_width), 1, 1])
        anchors = torch.concat([centers, dims], dim=-1)

        return torch.reshape(anchors, [int(feature_height * feature_width * self._num_anchors), 4])

    def get_anchors(self, image_height, image_width):
        image_height = torch.tensor(image_height)
        image_width = torch.tensor(image_width)
        anchors = [
            self._get_anchors(
                torch.ceil(image_height / (2**i)),
                torch.ceil(image_width / (2**i)),
                i
            )
            for i in range(3, 8)
        ]
        return torch.concat(anchors, dim=0)

    def get_anchors_hr(self):
        anchors = dict()
        for i in range(len(self._anchor_dims)):
            anchors['P' + str(i + 3)] = torch.ceil(self._anchor_dims[i]).type(dtype=torch.int16).squeeze(dim=0).squeeze(dim=0)
        return anchors


if __name__ == '__main__':
    make_anchorbox = AnchorBox()

    anchor_boxes =  make_anchorbox.get_anchors(torch.tensor(416), torch.tensor(416))

    print(anchor_boxes.shape)
    print(anchor_boxes)
