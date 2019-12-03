# uncompyle6 version 3.5.1
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Sep  7 2019, 18:27:02) 
# [Clang 10.0.1 (clang-1001.0.46.4)]
# Embedded file name: /root/divisioin_test/Fractional_segmentation.py
# Compiled at: 2019-10-25 11:41:56
# Size of source mod 2**32: 24707 bytes
"""
Created on Mon Oct 14 14:18:22 2019

@author: JRW
"""
import os, cv2, copy, time, pandas as pd, numpy as np
from PIL import Image
from skimage import morphology

def image_array(img):
    ret, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return binary


def division(img):
    img_source = preprocessed_image(img)
    img = cv2.normalize(img_source, None, 0, 1, (cv2.NORM_MINMAX), dtype=(cv2.CV_8U))
    img = 1 - img
    skeleton = morphology.skeletonize(img)
    skeleton = (1 - skeleton) * 255
    skeleton = skeleton.astype(np.uint8)
    img_1_part = Image.fromarray(cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB))
    img_1_part_temp = Image.new('RGB', img_1_part.size, '#ffffff')
    img_1_part_temp.paste(img_1_part.crop((0, img_1_part.size[1] // 4, img_1_part.size[0], img_1_part.size[1] // 3 * 2)), (
     0, img_1_part.size[1] // 4, img_1_part.size[0], img_1_part.size[1] // 3 * 2))
    img_1 = cv2.cvtColor(np.asarray(img_1_part_temp), cv2.COLOR_BGR2GRAY)
    binary = image_array(img_source)
    df = pd.DataFrame(binary)
    binary_1 = image_array(img_1)
    interval = binary_1.shape[1] // 10
    partition_skeleton = determine_split_skeleton(interval, binary_1)
    if len(partition_skeleton) == 0:
        return ([], [])
    else:
        img_result_1 = copy.deepcopy(img_source)
        img_result_2 = copy.deepcopy(img_source)
        skeleton_interval = 0
        for i in range(len(partition_skeleton) // 4):
            skeleton_interval = skeleton_interval + find_white_pixel(df, partition_skeleton[i])

        skeleton_interval = skeleton_interval // (len(partition_skeleton) // 4)
        img_result_1_boundary = []
        for i in range(len(partition_skeleton)):
            img_result_1_boundary.append(find_boundary(df, (partition_skeleton[i]), skeleton_interval, mode=1))

        for i in range(len(img_result_1_boundary)):
            img_result_1[img_result_1_boundary[i][0]:, img_result_1_boundary[i][1]] = 255

        img_result_1[img_result_1_boundary[0][0]:, :img_result_1_boundary[0][1]] = 255
        img_result_1[img_result_1_boundary[(len(img_result_1_boundary) - 1)][0]:, img_result_1_boundary[(len(img_result_1_boundary) - 1)][1]:] = 255
        skeleton_interval = 0
        for i in range(len(partition_skeleton) // 4):
            skeleton_interval = skeleton_interval + find_white_pixel(df, (partition_skeleton[i]), mode=2)

        skeleton_interval = skeleton_interval // (len(partition_skeleton) // 4)
        img_result_2_boundary = []
        for i in range(len(partition_skeleton)):
            img_result_2_boundary.append(find_boundary(df, (partition_skeleton[i]), skeleton_interval, mode=2))

        for i in range(len(img_result_2_boundary)):
            img_result_2[:img_result_2_boundary[i][0], img_result_2_boundary[i][1]] = 255

        img_result_2[:img_result_2_boundary[0][0], :img_result_2_boundary[0][1]] = 255
        img_result_2[:img_result_2_boundary[(len(img_result_2_boundary) - 1)][0], img_result_2_boundary[(len(img_result_2_boundary) - 1)][1]:] = 255
        binary_img_1 = image_array(img_result_1)
        binary_img_2 = image_array(img_result_2)
        location_img_1 = confirm_tailor_coordinates(binary_img_1)
        location_img_2 = confirm_tailor_coordinates(binary_img_2)
        img_result_1 = img_result_1[location_img_1[0][1]:location_img_1[1][1], location_img_1[0][0]:location_img_1[1][0]]
        img_result_2 = img_result_2[location_img_2[0][1]:location_img_2[1][1], location_img_2[0][0]:location_img_2[1][0]]
        return (
         img_result_1, img_result_2)


def find_boundary(df, loc, skeleton_interval, mode=1):
    num = 0
    index = loc[0]
    column = loc[1]
    if mode == 1:
        while df.iloc[index][column] == 255:
            num += 1
            index -= 1
            if num >= 3 * skeleton_interval:
                return [loc[0] - skeleton_interval, loc[1]]

        return [
         loc[0] - num + 1, loc[1]]
    else:
        while df.iloc[index][column] == 255:
            num += 1
            index += 1
            if num >= 3 * skeleton_interval:
                return [loc[0] + skeleton_interval, loc[1]]

        return [
         loc[0] + num, loc[1]]


def find_white_pixel(df, loc, mode=1):
    num = 0
    index = loc[0]
    column = loc[1]
    if mode == 1:
        while 1:
            if df.iloc[index][column] == 255:
                num += 1
                index -= 1

        return num
    else:
        while 1:
            if df.iloc[index][column] == 255:
                num += 1
                index += 1

        return num


def preprocessed_image(img):
    img_source = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    img_source = img_source.resize((int(img_source.size[0] / img_source.size[1] * 100), 100), Image.ANTIALIAS)
    img = cv2.cvtColor(np.asarray(img_source), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    return binary


def determine_split_skeleton(interval, binary):
    df = pd.DataFrame(binary)
    list_shuhe = list(df.sum(axis=0))
    interval = df.shape[1] // 4
    for i in range(len(list_shuhe) - 1, -1, -1):
        if list_shuhe[i] != 0:
            termination_num = i
            break

    for i in range(len(list_shuhe)):
        if list_shuhe[i] != 0:
            all_begin_pixel_index = i
            break

    pixel_index = all_begin_pixel_index
    temp = list(df[df[(pixel_index + 1)].isin([255])].index.values)
    if len(temp) == 0:
        return []
    else:
        continuity_points = continuity_detection(temp)
        basic_point, basic_point_pool = determine_initial_endpoint(df, pixel_index, continuity_points)
        count = []
        all_path_point = []
        for i in range(len(basic_point_pool)):
            count.append(1)
            all_path_point.append([1, basic_point_pool[i]])

        while pixel_index < termination_num:
            pixel_index += 1
            temp = list(df[df[pixel_index].isin([255])].index.values)
            if len(temp) == 0:
                continue
            continuity_points = continuity_detection(temp)
            for i in range(len(continuity_points)):
                if len(continuity_points[i]) != 1:
                    if pixel_position_feedback(df, [continuity_points[i][0], pixel_index]) == 1:
                        pass
                if confirm_the_following(df, [continuity_points[i][0], pixel_index]):
                    same_column_flag, temp = detection_with_column([continuity_points[i][0], pixel_index], all_path_point)
                    if not same_column_flag:
                        path_num = find_conjunction(temp, [continuity_points[i][0] - 1, pixel_index - 1], all_path_point)
                        if path_num > len(all_path_point):
                            all_path_point.append([1, [continuity_points[i][0] - 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                        else:
                            all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                    else:
                        all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                        path_num = find_conjunction(temp, [continuity_points[i][0] - 1, pixel_index - 1], all_path_point)
                        if path_num > len(all_path_point):
                            all_path_point.append([1, [continuity_points[i][0] - 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                        else:
                            all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                if pixel_position_feedback(df, [continuity_points[i][0], pixel_index]) == 2:
                    if confirm_the_following(df, [continuity_points[i][0], pixel_index]):
                        same_column_flag, temp = detection_with_column([continuity_points[i][0], pixel_index], all_path_point)
                        if not same_column_flag:
                            path_num = find_conjunction(temp, [continuity_points[i][0], pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0], pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                        else:
                            all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                            path_num = find_conjunction(temp, [continuity_points[i][0], pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0], pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                    if pixel_position_feedback(df, [continuity_points[i][0], pixel_index]) == 3 and confirm_the_following(df, [continuity_points[i][0], pixel_index]):
                        same_column_flag, temp = detection_with_column([continuity_points[i][0], pixel_index], all_path_point)
                        if not same_column_flag:
                            path_num = find_conjunction(temp, [continuity_points[i][0] + 1, pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0] + 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                        else:
                            all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                            path_num = find_conjunction(temp, [continuity_points[i][0] + 1, pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0] + 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                    if pixel_position_feedback(df, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]) == 1:
                        if confirm_the_following(df, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]):
                            same_column_flag, temp = detection_with_column([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index], all_path_point)
                            if not same_column_flag:
                                path_num = find_conjunction(temp, [continuity_points[i][(len(continuity_points[i]) - 1)] - 1, pixel_index - 1], all_path_point)
                                if path_num > len(all_path_point):
                                    all_path_point.append([1, [continuity_points[i][(len(continuity_points[i]) - 1)] - 1, pixel_index - 1], [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]])
                                else:
                                    all_path_point[path_num].append([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index])
                            else:
                                all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                                path_num = find_conjunction(temp, [continuity_points[i][(len(continuity_points[i]) - 1)] - 1, pixel_index - 1], all_path_point)
                                if path_num > len(all_path_point):
                                    all_path_point.append([1, [continuity_points[i][(len(continuity_points[i]) - 1)] - 1, pixel_index - 1], [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]])
                                else:
                                    all_path_point[path_num].append([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index])
                    if pixel_position_feedback(df, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]) == 2:
                        if confirm_the_following(df, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]):
                            same_column_flag, temp = detection_with_column([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index], all_path_point)
                            if not same_column_flag:
                                path_num = find_conjunction(temp, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index - 1], all_path_point)
                                if path_num > len(all_path_point):
                                    all_path_point.append([1, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index - 1], [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]])
                                else:
                                    all_path_point[path_num].append([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index])
                            else:
                                all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                                path_num = find_conjunction(temp, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index - 1], all_path_point)
                                if path_num > len(all_path_point):
                                    all_path_point.append([1, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index - 1], [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]])
                                else:
                                    all_path_point[path_num].append([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index])
                    if pixel_position_feedback(df, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]) == 3:
                        if confirm_the_following(df, [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]):
                            same_column_flag, temp = detection_with_column([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index], all_path_point)
                            if not same_column_flag:
                                path_num = find_conjunction(temp, [continuity_points[i][(len(continuity_points[i]) - 1)] + 1, pixel_index - 1], all_path_point)
                                if path_num > len(all_path_point):
                                    all_path_point.append([1, [continuity_points[i][(len(continuity_points[i]) - 1)] + 1, pixel_index - 1], [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]])
                                else:
                                    all_path_point[path_num].append([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index])
                            else:
                                all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                                path_num = find_conjunction(temp, [continuity_points[i][(len(continuity_points[i]) - 1)] + 1, pixel_index - 1], all_path_point)
                                if path_num > len(all_path_point):
                                    all_path_point.append([1, [continuity_points[i][(len(continuity_points[i]) - 1)] + 1, pixel_index - 1], [continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index]])
                                else:
                                    all_path_point[path_num].append([continuity_points[i][(len(continuity_points[i]) - 1)], pixel_index])
                    elif pixel_position_feedback(df, [continuity_points[i][0], pixel_index]) == 1:
                        same_column_flag, temp = detection_with_column([continuity_points[i][0], pixel_index], all_path_point)
                        if not same_column_flag:
                            path_num = find_conjunction(temp, [continuity_points[i][0] - 1, pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0] - 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                        else:
                            all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                            path_num = find_conjunction(temp, [continuity_points[i][0] - 1, pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0] - 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                    elif pixel_position_feedback(df, [continuity_points[i][0], pixel_index]) == 2:
                        same_column_flag, temp = detection_with_column([continuity_points[i][0], pixel_index], all_path_point)
                        if not same_column_flag:
                            path_num = find_conjunction(temp, [continuity_points[i][0], pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0], pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                        else:
                            all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                            path_num = find_conjunction(temp, [continuity_points[i][0], pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0], pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                    elif pixel_position_feedback(df, [continuity_points[i][0], pixel_index]) == 3:
                        same_column_flag, temp = detection_with_column([continuity_points[i][0], pixel_index], all_path_point)
                        if not same_column_flag:
                            path_num = find_conjunction(temp, [continuity_points[i][0] + 1, pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0] + 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])
                        else:
                            all_path_point.append(all_path_point[temp][:len(all_path_point[temp]) - 1])
                            path_num = find_conjunction(temp, [continuity_points[i][0] + 1, pixel_index - 1], all_path_point)
                            if path_num > len(all_path_point):
                                all_path_point.append([1, [continuity_points[i][0] + 1, pixel_index - 1], [continuity_points[i][0], pixel_index]])
                            else:
                                all_path_point[path_num].append([continuity_points[i][0], pixel_index])

        first_screening = []
        for i in range(len(all_path_point)):
            if len(all_path_point[i]) >= interval:
                first_screening.append(all_path_point[i][1:])

        if len(first_screening) == 0:
            return []
        num = []
        for i in range(len(first_screening)):
            temp = 0
            temp = temp + len(first_screening[i]) * 0.6 + statistical_white_pixel(df, first_screening[i][0]) * 10
            num.append(temp)

        return first_screening[num.index(max(num))]


def statistical_white_pixel(df, loc):
    num = 8
    if loc[1] != 0:
        temp = [
         [
          loc[0] - 1, loc[1] - 1], [loc[0] - 1, loc[1]], [loc[0] - 1, loc[1] + 1],
         [
          loc[0], loc[1] - 1], [loc[0], loc[1] + 1],
         [
          loc[0] + 1, loc[1] - 1], [loc[0] + 1, loc[1]], [loc[0] + 1, loc[1] + 1]]
    else:
        temp = [
         [
          loc[0] - 1, loc[1]], [loc[0] - 1, loc[1] + 1],
         [
          loc[0], loc[1] + 1],
         [
          loc[0] + 1, loc[1]], [loc[0] + 1, loc[1] + 1]]
    for i in range(len(temp)):
        if df.iloc[temp[i][0]][temp[i][1]] == 255:
            num -= 1

    return num


def find_conjunction(same_columns, coordinate, all_path_point):
    for i in range(len(all_path_point)):
        if i == same_columns:
            continue
        if all_path_point[i][0] == 1:
            pass
        if coordinate == all_path_point[i][(len(all_path_point[i]) - 1)]:
            return i
            continue

    return len(all_path_point) + 1


def confirm_the_following(df, loc):
    if df.iloc[loc[0]][(loc[1] + 1)] == 255 or df.iloc[(loc[0] - 1)][(loc[1] + 1)] == 255 or df.iloc[(loc[0] + 1)][(loc[1] + 1)] == 255:
        return True
    else:
        return False


def detection_with_column(coordinate, all_path_point):
    for i in range(len(all_path_point)):
        if all_path_point[i][0] == 1:
            pass
        if coordinate[1] == all_path_point[i][(len(all_path_point[i]) - 1)][1]:
            if determine_relationship(all_path_point[i][(len(all_path_point[i]) - 2)], coordinate):
                return (True, i)
            else:
                continue

    return (False, -1)


def determine_relationship(loc, loc1):
    if loc[0] == loc1[0] and loc[1] + 1 == loc1[1] or loc[0] - 1 == loc1[0] and loc[1] + 1 == loc1[1] or loc[0] + 1 == loc1[0] and loc[1] + 1 == loc1[1]:
        return True
    else:
        return False


def determine_initial_endpoint(df, pixel_index, continuity_points):
    basic_point = []
    basic_point_pool = []
    pixel_index += 1
    for i in range(len(continuity_points)):
        for j in range(len(continuity_points[i])):
            if pixel_position_feedback(df, [continuity_points[i][j], pixel_index]) == 1:
                basic_point_pool.append([continuity_points[i][j] - 1, pixel_index - 1])
                break
            elif pixel_position_feedback(df, [continuity_points[i][j], pixel_index]) == 2:
                basic_point_pool.append([continuity_points[i][j], pixel_index - 1])
                break
            elif pixel_position_feedback(df, [continuity_points[i][j], pixel_index]) == 3:
                basic_point_pool.append([continuity_points[i][j] + 1, pixel_index - 1])
                break

    return (basic_point, basic_point_pool)


def pixel_position_feedback(df, loc):
    if df.iloc[loc[0]][(loc[1] - 1)] == 255:
        return 2
    elif df.iloc[(loc[0] - 1)][(loc[1] - 1)] == 255:
        return 1
    elif df.iloc[(loc[0] + 1)][(loc[1] - 1)] == 255:
        return 3
    else:
        return 0


def slope():
    print('ok')


def continuity_detection(index_list):
    initialization = index_list[0]
    continuity_points = [[]]
    count = 0
    for i in range(len(index_list)):
        if index_list[i] - initialization < 2:
            continuity_points[count].append(index_list[i])
            initialization = index_list[i]
        else:
            count += 1
            continuity_points.append([])
            continuity_points[count].append(index_list[i])
            initialization = index_list[i]

    return continuity_points


def confirm_tailor_coordinates(binary):
    location = []
    df = pd.DataFrame(binary)
    list_shuhe = list(df.sum(axis=1))
    list_he = list(df.sum(axis=0))
    for i in range(len(list_he)):
        if list_he[i] != 0:
            location.append([i])
            break

    for i in range(len(list_he) - 1, -1, -1):
        if list_he[i] != 0:
            location.append([i])
            break

    for i in range(len(list_shuhe)):
        if list_shuhe[i] != 0:
            location[0].append(i)
            break

    for i in range(len(list_shuhe) - 1, -1, -1):
        if list_shuhe[i] != 0:
            location[1].append(i)
            break

    return location
# okay decompiling Fractional_segmentation.pyc
