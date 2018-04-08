import numpy as np
import scipy.io 
import os 
import matplotlib.pyplot as plt
data = 'biosig'# 'farsiu' or 'biosig'
if data  in 'farsiu':
    filepath = 'F:\Hamed\postdoc\livewire\OcularData\\2015_BOE_Chiu'
    individual_train = range(1,9);   individual_test = [9,10];
    real_num_images_per_indiv = 11;     num_images_per_indiv = 11
    wrap_val= 10
    depth = 8
else:
    individual_train = range(1,11);   individual_test = [11,12,13];
    real_num_images_per_indiv = 10;     num_images_per_indiv = 10
    wrap_val= 9
    filepath = 'F:\Hamed\postdoc\livewire\OcularData'
    depth = 16
#filepath = '/scratch/suj-571-aa/myfiles/new_experiment/livewire/OcularData/2015_BOE_Chiu'
filepath1 = filepath

os.chdir(filepath)
weight1 = 10.
weight2 = 5.
IMAGE_SIZE1 = 128  # 326
IMAGE_SIZE2 = 128  # 466
# os.system('cd %s'%(os.path.join(filepath,'data1')))
num_images_per_indiv = real_num_images_per_indiv + wrap_val
num_3d_channels = 1
num_images_train = num_images_per_indiv * len(individual_train)
coef = 1
image_total = np.zeros((coef * num_images_train, IMAGE_SIZE1 * IMAGE_SIZE2));
annotation_total = np.zeros((coef * num_images_train, IMAGE_SIZE1 * IMAGE_SIZE2));
weights_total = np.zeros((coef * num_images_train, IMAGE_SIZE1 * IMAGE_SIZE2));
for indiv in individual_train:
    image = scipy.io.loadmat(os.path.join(filepath1, 'Subject_%.2d_128_cropped_resize.mat' % (indiv)))
    image_original = image['images_resize']
    # plt.imshow(image_original[:,:,4].astype('uint8'))
    # plt.show()
    image = np.transpose(image_original, (2, 0, 1))
    image = np.reshape(image, (real_num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    image = np.concatenate((image, image[0:wrap_val, :]), axis=0)
    image_total[(indiv - individual_train[0]) * num_images_per_indiv * coef:(indiv - individual_train[
        0] + 1) * num_images_per_indiv * coef, :] = image;  # np.concatenate((image,image),axis=0)
    annotation = scipy.io.loadmat('Subject_%.2d_128_cropped_resize_grndtrth.mat' % (indiv))
    annotation1_original = annotation['images_grndtrth1_resize']
    annotation1 = np.transpose(annotation1_original, (2, 0, 1))
    annotation1 = np.reshape(annotation1, (real_num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    annotation1 = np.concatenate((annotation1, annotation1[0:wrap_val, :]), axis=0)
    #    annotation2_original = annotation['images_grndtrth2_resize']
    #    annotation2 = np.transpose(annotation2_original, (2,0,1))
    #    annotation2 = np.reshape(annotation2,(real_num_images_per_indiv,IMAGE_SIZE1*IMAGE_SIZE2))
    #    annotation2 = np.concatenate((annotation2,annotation2[0:wrap_val,:]),axis=0)
    annotation_total[(indiv - individual_train[0]) * num_images_per_indiv * coef:(indiv - individual_train[
        0] + 1) * num_images_per_indiv * coef, :] = annotation1  # np.concatenate((annotation1,annotation2),axis=0)
    weights_matrix1 = np.zeros((num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    #    weights_matrix2 = np.zeros((num_images_per_indiv,IMAGE_SIZE1*IMAGE_SIZE2))

    gx, gy, gz = np.gradient(annotation1_original)
    grad_annotated = np.sqrt(np.square(gx) + np.square(gy) + np.square(gz))
    grad_annotated = grad_annotated / np.max(grad_annotated)
    output_grad_annotated1 = grad_annotated > 0

    #    gx, gy, gz = np.gradient(annotation2_original)
    #    grad_annotated = np.sqrt(np.square(gx)+np.square(gy)+np.square(gz))
    #    grad_annotated = grad_annotated/np.max(grad_annotated)
    #    output_grad_annotated2 = grad_annotated>0

    weights_single = 1 + weight1 * output_grad_annotated1 + weight2 * (
    np.logical_and(annotation1_original != 0, annotation1_original != 8))
    weights_matrix1 = np.transpose(weights_single, (2, 0, 1))
    weights_matrix1 = np.reshape(weights_matrix1, (real_num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    weights_matrix1 = np.concatenate((weights_matrix1, weights_matrix1[0:wrap_val, :]), axis=0)

    #    weights_single = 1 + weight1*output_grad_annotated2 + weight2* (np.logical_and(annotation2_original!=0 , annotation2_original!=8))
    #    weights_matrix2 = np.transpose(weights_single, (2,0,1))
    #    weights_matrix2 = np.reshape(weights_matrix2,(real_num_images_per_indiv,IMAGE_SIZE1*IMAGE_SIZE2))
    #    weights_matrix2 = np.concatenate((weights_matrix2,weights_matrix2[0:wrap_val,:]),axis=0)

    weights_total[(indiv - individual_train[0]) * num_images_per_indiv * coef:(indiv - individual_train[
        0] + 1) * num_images_per_indiv * coef,
    :] = weights_matrix1  # np.concatenate((weights_matrix1,weights_matrix2),axis=0)

image_total = np.reshape(image_total, [coef * num_3d_channels * len(individual_train),
                                       int(num_images_per_indiv * IMAGE_SIZE1 * IMAGE_SIZE2 / num_3d_channels)])
annotation_total = np.reshape(annotation_total, [coef * num_3d_channels * len(individual_train), int(
    num_images_per_indiv * IMAGE_SIZE1 * IMAGE_SIZE2 / num_3d_channels)])
weights_total = np.reshape(weights_total, [coef * num_3d_channels * len(individual_train),
                                             int(num_images_per_indiv * IMAGE_SIZE1 * IMAGE_SIZE2 / num_3d_channels)])

outdata = np.concatenate((image_total, annotation_total, weights_total), axis=1)
outdata = outdata.astype('uint8')
outdata.tofile(data+'_OCT_train_weighted_cropped_cycled_3d_128.bin')
print(np.shape(outdata))
print('save train data')
num_images_per_indiv = real_num_images_per_indiv
if depth > real_num_images_per_indiv:
    num_images_per_indiv = depth
num_images_test = num_images_per_indiv * len(individual_test)
image_total = np.zeros((num_images_test, IMAGE_SIZE1 * IMAGE_SIZE2));
annotation_total = np.zeros((num_images_test, IMAGE_SIZE1 * IMAGE_SIZE2));
weights_total = np.zeros((num_images_test, IMAGE_SIZE1 * IMAGE_SIZE2));

for indiv in individual_test:
    image = scipy.io.loadmat(os.path.join(filepath1, 'Subject_%.2d_128_cropped_resize.mat' % (indiv)))
    image = image['images_resize']
    image = np.transpose(image, (2, 0, 1))
    image = np.reshape(image, (real_num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    if depth>real_num_images_per_indiv:
        image = np.concatenate((image, image[0:(depth-real_num_images_per_indiv), :]), axis=0)
    image_total[(indiv - individual_test[0]) * num_images_per_indiv:(indiv - individual_test[0] + 1) * num_images_per_indiv,:] = image
    annotation = scipy.io.loadmat('Subject_%.2d_128_cropped_resize_grndtrth.mat' % (indiv))
    annotation1_original = annotation['images_grndtrth1_resize']
    annotation1 = np.transpose(annotation1_original, (2, 0, 1))
    annotation1 = np.reshape(annotation1, (real_num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    if depth>real_num_images_per_indiv:
        annotation1 = np.concatenate((annotation1, annotation1[0:(depth-real_num_images_per_indiv), :]), axis=0)
    #    annotation2_oiginal = annotation['images_grndtrth2']
    #    annotation2 = np.transpose(annotation2, (2,0,1))
    #    annotation2 = np.reshape(annotation2,(11,IMAGE_SIZE1*IMAGE_SIZE2))
    annotation_total[
    (indiv - individual_test[0]) * num_images_per_indiv:(indiv - individual_test[0] + 1) * num_images_per_indiv,
    :] = annotation1

    gx, gy, gz = np.gradient(annotation1_original)
    grad_annotated = np.sqrt(np.square(gx) + np.square(gy) + np.square(gz))
    grad_annotated = grad_annotated / np.max(grad_annotated)
    output_grad_annotated1 = grad_annotated > 0

    weights_single = 1 + weight1 * output_grad_annotated1 + weight2 * (
    np.logical_and(annotation1_original != 0, annotation1_original != 8))
    weights_matrix1 = np.transpose(weights_single, (2, 0, 1))
    weights_matrix1 = np.reshape(weights_matrix1, (real_num_images_per_indiv, IMAGE_SIZE1 * IMAGE_SIZE2))
    if depth>real_num_images_per_indiv:
        weights_matrix1 = np.concatenate((weights_matrix1, weights_matrix1[0:(depth-real_num_images_per_indiv), :]), axis=0)
    weights_total[
    (indiv - individual_test[0]) * num_images_per_indiv:(indiv - individual_test[0] + 1) * num_images_per_indiv,
    :] = weights_matrix1

image_total = np.reshape(image_total, [num_3d_channels * len(individual_test),
                                       int(num_images_per_indiv * IMAGE_SIZE1 * IMAGE_SIZE2 / num_3d_channels)])
annotation_total = np.reshape(annotation_total, [num_3d_channels * len(individual_test), int(
    num_images_per_indiv * IMAGE_SIZE1 * IMAGE_SIZE2 / num_3d_channels)])
weights_total = np.reshape(weights_total, [num_3d_channels * len(individual_test),
                                             int(num_images_per_indiv * IMAGE_SIZE1 * IMAGE_SIZE2 / num_3d_channels)])

print(np.shape(image_total))

image_total_cycled = [
    image_total[:, range(((cyc - depth) * IMAGE_SIZE1 * IMAGE_SIZE2), cyc * IMAGE_SIZE1 * IMAGE_SIZE2)] for cyc in
    range(real_num_images_per_indiv)]
annotation_total_cycled = [
    annotation_total[:, range(((cyc - depth) * IMAGE_SIZE1 * IMAGE_SIZE2), cyc * IMAGE_SIZE1 * IMAGE_SIZE2)] for cyc in
    range(real_num_images_per_indiv)]
weights_total_cycled = [
    weights_total[:, range(((cyc - depth) * IMAGE_SIZE1 * IMAGE_SIZE2), cyc * IMAGE_SIZE1 * IMAGE_SIZE2)] for cyc in
    range(real_num_images_per_indiv)]

image_total_cycled = np.concatenate(image_total_cycled, axis=0)
annotation_total_cycled = np.concatenate(annotation_total_cycled, axis=0)
weights_total_cycled = np.concatenate(weights_total_cycled, axis=0)

outdata = np.concatenate((image_total_cycled, annotation_total_cycled, weights_total_cycled), axis=1)
print(np.shape(outdata))
outdata = outdata.astype('uint8')
outdata.tofile(data+'_OCT_test_weighted_cropped_cycled_3d_128_'+str(depth)+'.bin')
print('save test data')