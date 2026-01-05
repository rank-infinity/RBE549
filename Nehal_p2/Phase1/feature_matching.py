import cv2
import numpy as np

data_path = "Nehal_p2/Phase1/P2Data"


class Image:
    def __init__(self, img_idx, total_images):
        self.img_idx = img_idx
        self.total_images = total_images
        self.image = cv2.imread(data_path+"/"+str(img_idx)+".png")

        self.colors = {}
        self.matches= [[] for _ in range(total_images)]
        self.match_ids = [i for i in range(1,total_images + 1)]

        matches_file = data_path+"/matching"+str(img_idx)+"_dedup.txt"
        self.fillMatches(matches_file)

    def getMatches(self, matched_img_idx):
        index = self.match_ids.index(matched_img_idx)
        arr  = np.array(self.matches[index]).T
        return arr

    def fillMatches(self, matches_file):
        file = self.createListOfNumbers(matches_file)
        for line in file:
            # Save color per feature in current image
            self.colors[(line[4], line[5])] = (line[1], line[2], line[3])

            # Save matches according to image_ids
            for i in range(int(line[0])-1):
                matched_img_idx = int(line[6 + i * 3])
                matched_img_x = line[7 + i * 3]
                matched_img_y = line[8 + i * 3]
                # print(matched_img_idx, matched_img_x, matched_img_y)
                self.matches[matched_img_idx-1].append((line[4], line[5],matched_img_x, matched_img_y))

    # numMatches, r, g, b, curr_img_x,  curr_img_y, matched_img_idx, matched_img_x, matched_img_y
    # 0            1  2  3    4          5           6               7               8
    def createListOfNumbers(self, matches_file):
        list_of_numbers = []
        with open(matches_file, 'r') as f:
            next(f, None)
            for line in f:
                numbers = list(map(float, line.split()))
                list_of_numbers.append(numbers)

        # print(list_of_numbers)
        return list_of_numbers


def read_Calibration_File(data_path):
    calibration_file = data_path + "/calibration.txt"
    with open(calibration_file, 'r') as f:
        lines = f.readlines()
        K_values = []
        for line in lines:
            K_values.extend(list(map(float, line.strip().split())))
        K = np.array(K_values).reshape(3, 3)
    print("Calibration Matrix K:\n", K)
    return K

if __name__ == "__main__":
    Image4 = Image(4, 5)
    print(len(Image4.matches[0]))  # Matches with Image 1
    print(len(Image4.matches[1]))  # Matches with Image 2   
    print(len(Image4.matches[2]))  # Matches with Image 3
    print(len(Image4.matches[3]))  # Matches with Image 4
    print(len(Image4.matches[4]))        # All matches
    # print(Image4.matches[4])   # First match with Image 5
    print(len(Image4.colors))      # Features in Image 5
