"""
DSC 20 Project Winter 2026
Name(s): Angelina Tran
PID(s):  A19292172
Sources: Python Docs, W3Schools, Stack Overflow
"""

import numpy as np
import os
from PIL import Image
import copy

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

# YOU SHOULD NOT MODIFY THESE TWO METHODS

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2

        >>> RGBImage([])
        Traceback (most recent call last):
        ...
        TypeError

        >>> RGBImage([1, 2, 3])
        Traceback (most recent call last):
        ...
        TypeError

        >>> RGBImage([[]])
        Traceback (most recent call last):
        ...
        TypeError

        >>> RGBImage([[[1, 2, 3]], [[4, 5, 6], [7, 8, 9]]])
        Traceback (most recent call last):
        ...
        TypeError

        >>> RGBImage([[[1, 2]], [[3, 4, 5]]])
        Traceback (most recent call last):
        ...
        TypeError

        >>> RGBImage([[[1, 2, 256]]])
        Traceback (most recent call last):
        ...
        ValueError

        >>> RGBImage([[[1, 2, -1]]])
        Traceback (most recent call last):
        ...
        ValueError

        >>> RGBImage([[[1, 2, '3']]])
        Traceback (most recent call last):
        ...
        ValueError
        """
        if not isinstance(pixels, list) or len(pixels) == 0:
            raise TypeError()
        if not all(isinstance(row, list) and len(row) > 0 for row in pixels):
            raise TypeError()
        if not all(len(row) == len(pixels[0]) for row in pixels):
            raise TypeError()
        if not all(isinstance(p, list) and len(p) == 3 
            for r in pixels for p in r):
            raise TypeError()
        if not all(isinstance(v, int) for r in pixels for p in r for v in p):
            raise ValueError()
        if any(v < 0 or v > 255 for r in pixels for p in r for v in p):
            raise ValueError()
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True

        >>> img = RGBImage([[[1, 2, 3]]])
        >>> pixels = img.get_pixels()
        >>> pixels[0][0][0] = 99
        >>> img.get_pixel(0, 0)
        (1, 2, 3)
        """
        new_pixels = [[[v for v in p] for p in r] for r in self.pixels]
        return new_pixels

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        >>> img = RGBImage([[[1, 2, 3]]])
        >>> pixels = img.get_pixels()
        >>> pixels[0][0][0] = 99
        >>> img.get_pixel(0, 0)
        (1, 2, 3)

        >>> img1 = RGBImage([[[10, 20, 30]]])
        >>> img2 = img1.copy()
        >>> img2.set_pixel(0, 0, (0, 0, 0))
        >>> img1.get_pixel(0, 0)
        (10, 20, 30)
        >>> img2.get_pixel(0, 0)
        (0, 0, 0)
        """
        new_pixels = self.get_pixels()
        return RGBImage(new_pixels)

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)

        >>> img = RGBImage([[[1, 2, 3], [4, 5, 6]]])

        >>> img.get_pixel(0, 1)
        (4, 5, 6)

        >>> img.get_pixel(-1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.get_pixel(0, -1)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.get_pixel('0', 0)
        Traceback (most recent call last):
        ...
        TypeError
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if row < 0 or row >= self.num_rows:
            raise ValueError()
        if col < 0 or col >= self.num_cols:
            raise ValueError()
        return tuple(self.pixels[row][col])


    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]

        >>> img = RGBImage([[[10, 20, 30], [40, 50, 60]]])
        >>> img.set_pixel(0, 0, (-1, 0, 100))
        >>> img.get_pixel(0, 0)
        (10, 0, 100)

        >>> img = RGBImage([[[10, 20, 30]]])
        >>> img.set_pixel(0, 0, (255, 255, 255))
        >>> img.get_pixel(0, 0)
        (255, 255, 255)

        >>> img = RGBImage([[[10, 20, 30]]])
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        >>> img = RGBImage([[[10, 20, 30]]])
        >>> img.set_pixel(0, 0, [1, 2, 3])
        Traceback (most recent call last):
        ...
        TypeError

        >>> img = RGBImage([[[10, 20, 30]]])
        >>> img.set_pixel(0, 0, (1, 2))
        Traceback (most recent call last):
        ...
        TypeError

        >>> img = RGBImage([[[10, 20, 30]]])
        >>> img.set_pixel(0, 0, (1, 2, '3'))
        Traceback (most recent call last):
        ...
        TypeError
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if row < 0 or row >= self.num_rows:
            raise ValueError()
        if col < 0 or col >= self.num_cols:
            raise ValueError()
        if not isinstance(new_color, tuple) or not len(new_color) == 3:
            raise TypeError()
        if not all(isinstance(v, int) for v in new_color):
            raise TypeError()
        if any(v > 255 for v in new_color):
            raise ValueError()
        self.pixels[row][col] = [
            new if new >= 0 else old
            for old, new in zip(self.pixels[row][col], new_color)
            ]

# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6

        >>> img = RGBImage([[[0, 128, 255]]])
        >>> proc = ImageProcessingTemplate()
        >>> out = proc.negate(img)
        >>> out.get_pixel(0, 0)
        (255, 127, 0)
        >>> img.get_pixel(0, 0)
        (0, 128, 255)
        """
        pixels = image.get_pixels()
        new_pixels = [[[255 - v for v in p] for p in r] for r in pixels]
        return RGBImage(new_pixels)
        

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)

        >>> img = RGBImage([[[1, 2, 5]]])
        >>> proc = ImageProcessingTemplate()
        >>> proc.grayscale(img).get_pixel(0, 0)
        (2, 2, 2)
        """
        pixels = image.get_pixels()
        new_pixels = [[[(sum(p)//3)] * 3 for p in r] for r in pixels]
        return RGBImage(new_pixels)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)

        >>> img = RGBImage([
        ... [[1, 1, 1], [2, 2, 2]],
        ... [[3, 3, 3], [4, 4, 4]]
        ... ])
        >>> proc = ImageProcessingTemplate()
        >>> out = proc.rotate_180(img)
        >>> out.get_pixel(0, 0)
        (4, 4, 4)
        >>> out.get_pixel(1, 1)
        (1, 1, 1)
        """
        pixels = image.get_pixels()
        reversed_rows = pixels[::-1]
        new_pixels = [row[::-1] for row in reversed_rows]
        return RGBImage(new_pixels)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86

        >>> img = RGBImage([[[100, 101, 101]]])
        >>> proc = ImageProcessingTemplate()
        >>> proc.get_average_brightness(img)
        100

        >>> img = RGBImage([
        ... [[0, 0, 0], [255, 255, 255]]
        ... ])
        >>> proc.get_average_brightness(img)
        127
        """
        pixels = image.get_pixels()
        pixel_brightness = [(sum(p)//3) for r in pixels for p in r ]
        return sum(pixel_brightness) // len(pixel_brightness)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 1.2)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)

        >>> img = RGBImage([[[100, 200, 250]]])
        >>> proc = ImageProcessingTemplate()
        >>> out = proc.adjust_brightness(img, 2.0)
        >>> out.get_pixel(0, 0)
        (200, 255, 255)

        >>> img = RGBImage([[[1, 10, 0]]])
        >>> proc.adjust_brightness(img, 0.25).get_pixel(0, 0)
        (0, 2, 0)

        >>> proc.adjust_brightness(img, 1)
        Traceback (most recent call last):
        ...
        TypeError
        """
        if not isinstance(intensity, float):
            raise TypeError()
        pixels = image.get_pixels()
        new_pixels = [
            [
                [max(0, min(255, int(v * intensity))) for v in p] 
                for p in r
            ] 
            for r in pixels
        ]
        return RGBImage(new_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = StandardImageProcessing()
        >>> img_proc.cost
        0

        >>> img = RGBImage([[[10, 20, 30]]])
        >>> proc = StandardImageProcessing()
        >>> proc.get_cost()
        0
        >>> _ = proc.negate(img)
        >>> proc.get_cost()
        5

        >>> proc = StandardImageProcessing()
        >>> proc.redeem_coupon(2)
        >>> _ = proc.negate(img)
        >>> _ = proc.grayscale(img)
        >>> proc.get_cost()
        0
        >>> _ = proc.rotate_180(img)
        >>> proc.get_cost()
        10

        >>> proc = StandardImageProcessing()
        >>> proc.redeem_coupon(2)
        >>> proc.redeem_coupon(3)
        >>> _ = proc.negate(img)
        >>> _ = proc.negate(img)
        >>> _ = proc.negate(img)
        >>> _ = proc.negate(img)
        >>> _ = proc.negate(img)
        >>> proc.get_cost()
        0

        >>> proc.redeem_coupon(0)
        Traceback (most recent call last):
        ...
        ValueError

        >>> proc.redeem_coupon(1.5)
        Traceback (most recent call last):
        ...
        TypeError
        """
        super().__init__()
        self.free_calls = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        result = super().negate(image)
        if self.free_calls > 0:
            self.free_calls -= 1
        else:
            self.cost += 5
        return result

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        result = super().grayscale(image)
        if self.free_calls > 0:
            self.free_calls -= 1
        else:
            self.cost += 6
        return result

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        result = super().rotate_180(image)
        if self.free_calls > 0:
            self.free_calls -= 1
        else:
            self.cost += 10
        return result

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        result = super().adjust_brightness(image, intensity)
        if self.free_calls > 0:
            self.free_calls -= 1
        else:
            self.cost += 1
        return result

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        self.free_calls += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        
        >>> proc = PremiumImageProcessing()
        >>> img = RGBImage([[[10, 20, 30]]])
        >>> _ = proc.negate(img)
        >>> _ = proc.grayscale(img)
        >>> proc.get_cost()
        50
        """
        self.cost = 50

    def pixelate(self, image, block_dim):
        """
        Returns a pixelated version of the image, where block_dim is the size of 
        the square blocks.

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_pixelate = img_proc.pixelate(img, 4)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_pixelate.png')
        >>> img_exp.pixels == img_pixelate.pixels # Check pixelate output
        True
        >>> img_save_helper('img/out/test_image_32x32_pixelate.png', img_pixelate)
        """
        pixels = image.get_pixels()
        new_pixels = image.get_pixels()
        num_rows = len(pixels)
        num_cols = len(pixels[0])
        for r in range(0, num_rows, block_dim):
            for c in range(0, num_cols, block_dim):
                block = []

                for i in range(r, r + block_dim):
                    for j in range(c, c + block_dim):
                        if i < num_rows and j < num_cols:
                            block.append(pixels[i][j])

                r_total = 0
                g_total = 0
                b_total = 0
                for pixel in block:
                    r_total += pixel[0]
                    g_total += pixel[1]
                    b_total += pixel[2]

                n = len(block)
                avg = [r_total // n, g_total // n, b_total // n]

                for i in range(r, r + block_dim):
                    for j in range(c, c + block_dim):
                        if i < num_rows and j < num_cols:
                            new_pixels[i][j] = avg
        return RGBImage(new_pixels)

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        pixels = image.get_pixels()
        new_pixels = image.get_pixels()
        num_rows = len(pixels)
        num_cols = len(pixels[0])
        gray = []
        for r in range(num_rows):
            row = []
            for c in range(num_cols):
                pixel = pixels[r][c]
                avg = (pixel[0] + pixel[1] + pixel[2]) // 3
                row.append(avg)
            gray.append(row)

        for r in range(num_rows):
            for c in range(num_cols):
                total = 0

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr = r + dr
                        nc = c + dc

                        if 0 <= nr < num_rows and 0 <= nc < num_cols:
                            if dr == 0 and dc == 0:
                                weight = 8
                            else:
                                weight = -1

                            total += gray[nr][nc] * weight
                total = max(0, min(255, total))
                new_pixels[r][c] = [total, total, total]
        return RGBImage(new_pixels)


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier

    >>> knn = ImageKNNClassifier(3)
    >>> img = RGBImage([[[1, 2, 3]]])
    >>> knn.fit([(img, 'cat'), (img, 'dog')])
    Traceback (most recent call last):
    ...
    ValueError

    >>> knn = ImageKNNClassifier(1)
    >>> knn.fit([(img, 'cat')])
    >>> len(knn.data)
    1
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors
        self.data = None 

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data) < self.k_neighbors:
            raise ValueError()

        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909

        >>> img1 = RGBImage([[[0, 0, 0]]])
        >>> img2 = RGBImage([[[0, 0, 0]]])
        >>> knn = ImageKNNClassifier(1)
        >>> knn.distance(img1, img2)
        0.0

        >>> img3 = RGBImage([[[3, 4, 0]]])
        >>> knn.distance(img1, img3)
        5.0

        >>> knn.distance(img1, [[[3, 4, 0]]])
        Traceback (most recent call last):
        ...
        TypeError

        >>> img4 = RGBImage([
        ... [[1, 2, 3], [4, 5, 6]]
        ... ])
        >>> knn.distance(img1, img4)
        Traceback (most recent call last):
        ...
        ValueError
        """
        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        if image1.size() != image2.size():
            raise ValueError()

        pixels1 = image1.get_pixels()
        pixels2 = image2.get_pixels()
        num_rows, num_cols = image1.size()

        total = sum([(pixels1[r][c][v] - pixels2[r][c][v]) ** 2 
        for r in range(num_rows) 
        for c in range(num_cols) 
        for v in range(3)]
        )

        return total ** 0.5

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['cat', 'cat', 'cat'])
        'cat'

        >>> knn.vote(['cat', 'dog', 'dog'])
        'dog'
        """
        return max(candidates, key=candidates.count)

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below

        >>> knn = ImageKNNClassifier(1)
        >>> img = RGBImage([[[1, 2, 3]]])
        >>> knn.predict(img)
        Traceback (most recent call last):
        ...
        ValueError

        >>> knn = ImageKNNClassifier(1)
        >>> img1 = RGBImage([[[0, 0, 0]]])
        >>> img2 = RGBImage([[[255, 255, 255]]])
        >>> knn.fit([(img1, 'dark'), (img2, 'bright')])
        >>> knn.predict(RGBImage([[[1, 1, 1]]]))
        'dark'
        """
        if self.data is None:
            raise ValueError()

        distances = [
            (self.distance(image, train_image), label)
            for train_image, label in self.data
        ]
        distances.sort()

        nearest = distances[:self.k_neighbors]
        candidates = [label for _, label in nearest]

        return self.vote(candidates)

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
