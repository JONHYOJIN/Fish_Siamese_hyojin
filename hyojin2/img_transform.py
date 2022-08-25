import cv2
import constant as const

def get_img(get_url, filename):
    address = get_url + filename
    img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(105, 105), interpolation=3)
    # cv2.imshow('Sample', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return img
def save_img(image, save_url, filename):
    address = save_url + filename
    cv2.imwrite(address, image)


if __name__=='__main__':
    #Train set
    # for sea in const.SEA:
    #     img = get_img(const.raw_url, sea)
    #     save_img(img, const.sea_url, sea)

    # for farm in const.FARM:
    #     img = get_img(const.raw_url, farm)
    #     save_img(img, const.farm_url, farm)

    #Test set
    # for sea in const.SEA_T:
    #     img = get_img(const.raw_test_url, sea)
    #     save_img(img, const.sea_test_url, sea)

    # for farm in const.FARM_T:
    #     img = get_img(const.raw_test_url, farm)
    #     save_img(img, const.farm_test_url, farm)

    #Validation set
    for sea in const.SEA_V:
        img = get_img(const.raw_val_url, sea)
        save_img(img, const.sea_val_url, sea)

    for farm in const.FARM_V:
        img = get_img(const.raw_val_url, farm)
        save_img(img, const.farm_val_url, farm)