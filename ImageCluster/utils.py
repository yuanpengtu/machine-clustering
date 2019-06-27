import shutil,os
def merge_imgs(root_dir):
    if not os.path.exists('train_data_2'):
        os.mkdir('train_data_2')
    for stage in os.listdir(root_dir):
        p=os.path.join(root_dir,stage)
        for j in os.listdir(p):
            i=os.path.join(p,j)
            for index,final in enumerate(os.listdir(i)):
                if j=='凹凸包':
                    j='aotubao'
                class_name = j + '_' + str(index)
                dir_path=os.path.join(i,final)
                if not os.path.exists(os.path.join('train_data_2', class_name)):
                    os.mkdir(os.path.join('train_data_2', class_name))
                for filename in os.listdir(dir_path):
                    shutil.copy(os.path.join(dir_path,filename), os.path.join('train_data_2', class_name))
                print(os.path.join('train_data_2',class_name)+'  Done')
if __name__=='__main__':
    merge_imgs('WR_data_Labor_20190414')
