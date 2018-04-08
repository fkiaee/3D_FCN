clc
clear all
close all
mfilename
layer_group1 = [1,7,8];
SOptions.dRadius = 350;
SOptions.dCaptureRadius = 4;
mypath = 'F:\Hamed\postdoc\livewire';
addpath(genpath(strcat(mypath,'\livewire_original')))
cd(strcat(mypath,'\OcularData'))
load(strcat(mypath,'\OcularData\randomImages_125665e61570be51a\randomImages\randomImages.mat'))
num_iter = 10;
num_individual = 13;
num_GA = 10;
%num_layers  = 1:12;
num_layers = [1,2,4,5,6,8,10,12];
% num_layers = [1,3,5,8,10];
yrange = 55:485;%val_min2 = 66.6; val_max2 = 473.7;
% load(strcat('D:\HAMEDFAHIMI\LiveWire\results\mae_vertical_truncated_constrained31_.mat'))
% hold on,
val_min2 = 800; val_max2 = 0;
for indiv = 11:num_individual
    indiv
    folder_name = ls(strcat('data',num2str(indiv),'_*'));
    cd(folder_name)
    cd(strcat('data',num2str(indiv)))
    load(num2str(indiv));%load(strcat(num2str(indiv),'_clean2.mat'));
    load(strcat('totalManual_',num2str(indiv)))
    images_new = zeros(length(yrange),481,10);     images_cartoon1 = zeros(length(yrange),481,10);
    images_resize = zeros(128,128,10);     images_cartoon1_resize = zeros(128,128,10);
    for Gold_id = 1:num_GA % on 10 images we have true answer
        im_id = random_images(indiv,Gold_id);
        dImg = d3(:,:,im_id);
        dImg_new = dImg;
        %         subplot(1,2,1); imshow(uint8(dImg));
        manualLayers1_new_prev = 1*ones(1,size(dImg,2));%size(dImg,1)*(0:(size(dImg,2)-1))+1;
        for layer_id = 1:length(num_layers);
            layer_id_real = num_layers(layer_id);
            manualLayers1_new = y_manual(layer_id_real,:,Gold_id)';
            %             hold on, plot(manualLayers1_new)
            X_non_NAN = 1:size(dImg,2);
            gg = X_non_NAN(isnan(manualLayers1_new));
            if ~isempty(gg)
                if gg(1)~=482 || gg(end)~=512
                    error('errroor')
                end
            end
            X_non_NAN = X_non_NAN(~isnan(manualLayers1_new));
            X_non_NAN = X_non_NAN(1:481);
            GA_non_NAN = manualLayers1_new(~isnan(manualLayers1_new));
            d1 = GA_non_NAN;
            if min(d1)<val_min2
                val_min2 = min(d1);
            end
            if max(d1)>val_max2
                val_max2 = max(d1);
            end
            ind1 = [];
            for xx = X_non_NAN
                ranges = round(manualLayers1_new_prev(xx)):round(manualLayers1_new(xx));
                ind1 = [ind1,sub2ind(size(dImg),ranges,xx*ones(1,length(ranges)))];
            end
            dImg_new(ind1) = layer_id-1;%round(median(dImg(ind1)));
            manualLayers1_new_prev = manualLayers1_new;
        end
        ind1 = [];
        for xx = X_non_NAN
            ranges = round(manualLayers1_new_prev(xx)):size(dImg,1);
            ind1 = [ind1,sub2ind(size(dImg),ranges,xx*ones(1,length(ranges)))];
        end
        dImg_new(ind1) = layer_id;%median(dImg(ind1));
        %         subplot(1,2,2); 
        images_new(:,:,Gold_id) = dImg(yrange,X_non_NAN);
        imshow(uint8(images_new(:,:,Gold_id)))
        images_cartoon1(:,:,Gold_id) = dImg_new(yrange,X_non_NAN);
        images_resize(:,:,Gold_id) = imresize(uint8(dImg(yrange,X_non_NAN)),[128,128]);
        temp = imresize(uint8(dImg_new(yrange,X_non_NAN)),[128,128]);
        temp(temp<0) = 0; temp(temp>8) = 8;
        images_cartoon1_resize(:,:,Gold_id) = temp;
%         set(gca(), 'LooseInset', get(gca(), 'TightInset'))
%         saveas(gca,strcat(mypath,'\images','\Img_fin_',num2str(indiv),'_',num2str(Gold_id),'_1.tif'))
%         hgexport(gcf,strcat(mypath,'\images','\Img_fin_',num2str(indiv),'_',num2str(Gold_id),'_1.tif'),hgexport('factorystyle'),'Format','jpeg')
    end
    cd ..
    cd ..
    manualLayers1_new = y_manual(num_layers,1:481,:)- yrange(1) + 1;
    save(strcat(sprintf('Subject_%.2d',indiv),'_128_cropped_resize.mat'),'images_new','images_resize','manualLayers1_new')
    save(strcat(sprintf('Subject_%.2d_128_cropped_resize',indiv),'_cartoon.mat'),'images_cartoon1','images_cartoon1_resize')

end