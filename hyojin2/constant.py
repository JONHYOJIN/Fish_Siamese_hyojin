#Train
SEA = ['sea1.jpeg','sea2.jpeg','sea3.jpeg','sea4.jpeg','sea5.jpeg',\
       'sea6.jpeg','sea7.jpeg','sea8.jpeg','sea9.jpeg','sea10.jpeg']
FARM = ['farm1.jpeg','farm2.jpeg','farm3.jpeg','farm4.jpeg','farm5.jpeg',\
        'farm6.jpeg','farm7.jpeg','farm8.jpeg','farm9.jpeg','farm10.jpeg']
ALL = ['sea1.jpeg','sea2.jpeg','sea3.jpeg','sea4.jpeg','sea5.jpeg',\
       'sea6.jpeg','sea7.jpeg','sea8.jpeg','sea9.jpeg','sea10.jpeg',\
       'farm1.jpeg','farm2.jpeg','farm3.jpeg','farm4.jpeg','farm5.jpeg',\
       'farm6.jpeg','farm7.jpeg','farm8.jpeg','farm9.jpeg','farm10.jpeg']
raw_url = './raw_flatfish_train/'
sea_url = './flatfish_train/sea/'
farm_url = './flatfish_train/farm/'

#Test
SEA_T = ['sea_t1.jpeg','sea_t2.jpeg','sea_t3.jpeg','sea_t4.jpeg','sea_t5.jpeg']
FARM_T = ['farm_t1.jpeg','farm_t2.jpeg','farm_t3.jpeg','farm_t4.jpeg','farm_t5.jpeg']

raw_test_url = './raw_flatfish_test/'
sea_test_url = './flatfish_test/sea/'
farm_test_url = './flatfish_test/farm/'

#Validation
SEA_V = ['sea_v1.jpeg','sea_v2.jpeg','sea_v3.jpeg','sea_v4.jpeg','sea_v5.jpeg']
FARM_V = ['farm_v1.jpeg','farm_v2.jpeg','farm_v3.jpeg','farm_v4.jpeg','farm_v5.jpeg']

raw_val_url = './raw_flatfish_val/'
sea_val_url = './flatfish_val/sea/'
farm_val_url = './flatfish_val/farm/'

#Others
BATCH_SIZE = 25
EPOCHS = 100
NUM_TRAIN = 400
NUM_VAL = 25
NUM_TEST = 25