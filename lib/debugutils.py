import numpy as np
import cv2

DEFAULT_RECT_COLOR = (255, 0, 0)  # B, G, R
ANNOTATE_MAX_TEXT_LEN = 20


def mark_rectangle(image, rect=None, text=None, color=DEFAULT_RECT_COLOR, line_width=2, draw_border=True, font_scale=0.5):
    """
    Marks a rectangle on top of an image (numpy array).
    """
    if not rect:
        rect = (0, 0, image.shape[1], image.shape[0])

    start_point = rect[0], rect[1]
    end_point = rect[0] + rect[2] - 1, rect[1] + rect[3] - 1

    if draw_border:
        image = cv2.rectangle(image, start_point, end_point, color, line_width)

    if text is not None:
        text_len = min(len(text), ANNOTATE_MAX_TEXT_LEN)
        cv2.putText(image, text[:text_len], (rect[0], rect[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    thickness=line_width)


def draw_cross(img, center, color, d, thickness=1, text=None):
    cv2.line(img, (center[0] - d, center[1]), (center[0] + d, center[1]), color, thickness)
    cv2.line(img, (center[0], center[1] - d), (center[0], center[1] + d), color, thickness)
    if text is not None:
        text_len = min(len(text), ANNOTATE_MAX_TEXT_LEN)
        cv2.putText(img, text[:text_len], (center[0], center[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def reg_image_to_rgb(reg_image, channel=2):
    """
    Convert regression image (float) to an rgb uint8 image. Values are colored blue.
    channel selects the channel to which to assign the (mono) values. valid valus are [-1,0,1,2], where -1 assigns to
    all channels.
    """
    allowed_channels = [-1, 0, 1, 2]
    if channel not in allowed_channels:
        raise ValueError(f'channel not in allowed channels: {allowed_channels}')

    v_min = np.min(reg_image)
    v_max = np.max(reg_image)
    if (v_max - v_min) == 0:
        return np.zeros(shape=(*reg_image.shape, 3), dtype='uint8')

    n_img = (reg_image - v_min) / (v_max - v_min) * 255
    n_img = n_img.astype('uint8')
    rgb_image = np.zeros(shape=(*reg_image.shape, 3), dtype='uint8')
    if channel == -1:
        # all channels (i.e. grayscale)
        for i in range(3):
            rgb_image[:, :, i] = n_img
    else:
        # a coloured channel (r/g/b)
        rgb_image[:, :, channel] = n_img

    return rgb_image

