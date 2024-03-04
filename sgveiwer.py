import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to draw lines on the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = np.dstack((img, img, img))
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    combined_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return combined_img

# longest two lines
def get_line_length(line):
    x1, y1, x2, y2 = line[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Function to calculate the intersection point of two lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
    return (px, py)


def convex_Hull(image,Cx,Cy,output_path):
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour is the arrow-like shape
    contour = max(contours, key=cv2.contourArea)

    # Find the convex hull of the contour
    hull = cv2.convexHull(contour)

    # Approximate the contour to simplify it
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Draw the convex hull and approximated contour
    image_with_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(image_with_hull, [hull], True, (0, 255, 0), 2)
    cv2.polylines(image_with_hull, [approx], True, (0, 0, 255), 2)

    farthest_dist = 0
    farthest_point = None
    for point in approx[:, 0, :]:
        dist = np.linalg.norm(point - image_center)
        if dist > farthest_dist:
            farthest_dist = dist
            farthest_point = tuple(point)

    # Draw the line from the image center to the farthest point
    cv2.line(image_with_hull, image_center, farthest_point, (255, 0, 0), 2)

    # Calculate the angle of the line with respect to the 12 o'clock position
    dx = farthest_point[0] - image_center[0]
    dy = image_center[1] - farthest_point[1]  # y coordinates go downward for images
    angle = (np.arctan2(dy, dx) * 180.0 / np.pi) % 360  # Convert radian to degree and ensure positive angle
    angle_from_12 = (90 - angle) % 360  # Adjust angle to the 12 o'clock reference

    image_with_text=put_text(image_with_hull, 'Angle: {:.2f}'.format(angle_from_12), (10, 30))

    cv2.imwrite(os.path.join(output_path, 'o0115A.png'), cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB)) 


#-------------HoughLinesP-------------------------
# Function to draw lines on the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = np.dstack((img, img, img))
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    combined_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return combined_img

# longest two lines
def get_line_length(line):
    x1, y1, x2, y2 = line[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Function to calculate the intersection point of two lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
    return (px, py)

def houghlines2(image,cx,cy,output_path):
    # HoughLinesP to find lines
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)


    # Draw the lines
    if lines is not None:
        img_with_lines = draw_lines(image, lines)
    else:
        img_with_lines = image.copy()


    # Sort the lines based on length
    sorted_lines = sorted(lines, key=get_line_length, reverse=True)
    # Take the longest two lines
    longest_two_lines = sorted_lines[:2]


    # Calculate the intersection point of the longest two lines
    intersection_point = line_intersection(longest_two_lines[0], longest_two_lines[1])

    # Draw the longest two lines
    img_with_longest_lines = draw_lines(image, longest_two_lines, color=[0, 255, 0], thickness=2)
    # Draw the line from the intersection point to the center of the image
    center_of_image = (332, 289)
    cv2.line(img_with_longest_lines, (int(intersection_point[0]), int(intersection_point[1])), center_of_image, [0, 0, 255], 2)

    # Save the result
    # output_path = 'D:\\git\\Deamnet\\index_2023_ver2_result0112a\\cc1115edge_with_lines.jpg'
    cv2.imwrite(os.path.join(output_path, '20240115_houghline.jpg'), cv2.cvtColor(img_with_longest_lines, cv2.COLOR_RGB2BGR))

    return img_with_longest_lines

#-------------end of HoughLinesP-------------------------



#-------------fit_lines------------------------------
def closest_points_to_center(contour, center):
    """找到轮廓上最接近中心点的两个点"""
    distances = np.sqrt((contour[:, :, 0] - center[0])**2 + (contour[:, :, 1] - center[1])**2)
    idx = np.argsort(distances, axis=0)[:2]
    return contour[idx].reshape(2, 2)

def draw_line_through_points(image, point1, point2, color=(0, 255, 0), thickness=2):
    """通过两点绘制直线"""
    cv2.line(image, tuple(point1), tuple(point2), color, thickness)


def fit_lines(image, xc, yc,output_path):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(image, 50, 150, apertureSize=3,L2gradient = True)
#     cv2.imwrite('D:\\git\\Deamnet\\index_2023_ver2_result0112a\\cc202302edge.jpg' , edges)
    #color_image

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    gray_image = cv2.imread("D:\git\Deamnet\sennsa\output_img_DNpoint_mask.jpg", cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 对每个轮廓找到最接近中心的两个点并绘制直线
    for contour in contours:
        if len(contour) >= 2:
            point1, point2 = closest_points_to_center(contour, (xc, yc))
            draw_line_through_points(color_image, point1, point2)
    cv2.imwrite(os.path.join(output_path, '20240115_fitlines.jpg'), color_image)

#-------------end of fit_lines------------------------


def put_text(image, text, point):
    """
    Add text to an image.
    Args:
        image (numpy.ndarray): The image to add text to.
        text (str): The text to be added.
        point (tuple): The coordinates of the starting point of the text.
    Returns:
        returns:       
        numpy.ndarray: The image with the text added. 
    """
    
    # Set the font, size, color, and thickness of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # white color
    thickness = 2

    # Get the width and height of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the x and y coordinates of the text box
    text_x = (image.shape[1] - text_width) // 2
    text_y = text_height + 10  # 10 pixels from the top

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    return(image)


# 1.Load the image

image_path = 'D:\git\Deamnet\sennsa\output_img_DNpoint_mask.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not loaded properly.")

image_center = (332, 289)
output_path="D:\\git\\Deamnet\\index_2023_ver2_result0115\\"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 2.convex_Hull
convex_Hull(image,image_center[0], image_center[1],output_path)

# 3.fit lines
fit_lines(image,image_center[0], image_center[1],output_path)

# 4.
houghlines2(image,image_center[0], image_center[1],output_path)



# # Load the image
# image_path = 'D:\\git\\Deamnet\\index_2023_ver2_result0112a\\cc202302edge.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # HoughLinesP to find lines
# lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)


# # Draw the lines
# if lines is not None:
#     img_with_lines = draw_lines(image, lines)
# else:
#     img_with_lines = image.copy()


# # Sort the lines based on length
# sorted_lines = sorted(lines, key=get_line_length, reverse=True)
# # Take the longest two lines
# longest_two_lines = sorted_lines[:2]



# # Calculate the intersection point of the longest two lines
# intersection_point = line_intersection(longest_two_lines[0], longest_two_lines[1])

# # Draw the longest two lines
# img_with_longest_lines = draw_lines(image, longest_two_lines, color=[0, 255, 0], thickness=2)

# # Draw the line from the intersection point to the center of the image
# center_of_image = (332, 289)
# cv2.line(img_with_longest_lines, (int(intersection_point[0]), int(intersection_point[1])), center_of_image, [0, 0, 255], 2)

# # Let's display the results
# plt.figure(figsize=(10, 10))
# plt.imshow(img_with_longest_lines)
# plt.show()

# # Save the result
# output_path = 'D:\\git\\Deamnet\\index_2023_ver2_result0112a\\cc202302edge_with_lines.jpg'
# cv2.imwrite(output_path, cv2.cvtColor(img_with_longest_lines, cv2.COLOR_RGB2BGR))

# output_path