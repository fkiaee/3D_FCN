clc
clear all
close all
mypath = 'F:\Hamed\postdoc\livewire';
addpath(genpath(strcat(mypath,'\livewire_original')))
cd(strcat(mypath,'\OcularData\2015_BOE_Chiu'))
num_individual = 10;
num_GA = 11;
num_layer  = 8;
image_size = 128;
% hold on,
h = figure; ax1 = axes(h);
CM = [jet(8);0,0,0];
for ff = 1:num_individual
    %     sprintf('%.2d',1)
    load(sprintf('Subject_%.2d_128_cropped_resize.mat',ff))
    images_grndtrth1 = zeros(size(images_new));%zeros(size(images_new));
    images_grndtrth2 = zeros(size(images_new));
    images_grndtrth1_resize = zeros(size(images_resize));%zeros(size(images_new));
    images_grndtrth2_resize = zeros(size(images_resize));
    for im_id = 1:size(images_new,3)
        dImg = images_new(:,:,im_id);
        dImg_new1 = zeros(size(dImg));
        dImg_new2 = zeros(size(dImg));
%         subplot(1,3,1); imshow(uint8(dImg));
        Gold_answer_prev1 = 1*ones(1,size(dImg,2)); Gold_answer_prev2 = 1*ones(1,size(dImg,2));%size(dImg,1)*(0:(size(dImg,2)-1))+1;
        for layer_id = 1:num_layer;            
            Gold_answer1 = manualLayers1_new(layer_id,:,im_id);
            if any(isnan(Gold_answer1))
                t = 1:numel(Gold_answer1);
                Gold_answer1 = interp1(t(~isnan(Gold_answer1)),Gold_answer1(~isnan(Gold_answer1)),t,'linear','extrap');
            end
%             hold on, plot(Gold_answer1)
            X_non_NAN = 1:size(dImg,2);
            GA_non_NAN = Gold_answer1(~isnan(Gold_answer1));
            ind1 = [];
            for xx = X_non_NAN
                ranges = round(max(Gold_answer_prev1(xx),1)):round(Gold_answer1(xx));
                ind1 = [ind1,sub2ind(size(dImg),ranges,xx*ones(1,length(ranges)))];
            end
            dImg_new1(ind1) = layer_id-1;%round(median(dImg(ind1)));
            Gold_answer_prev1 = Gold_answer1;
            
            
            Gold_answer2 = manualLayers1_new(layer_id,:,im_id);
            if any(isnan(Gold_answer2))
                t = 1:numel(Gold_answer2);
                Gold_answer2 = interp1(t(~isnan(Gold_answer2)),Gold_answer2(~isnan(Gold_answer2)),t,'linear','extrap');
            end
%             hold on, plot(Gold_answer2)
            X_non_NAN = 1:size(dImg,2);
            GA_non_NAN = Gold_answer2(~isnan(Gold_answer2));
            ind1 = [];
            for xx = X_non_NAN
                ranges = round(max(Gold_answer_prev2(xx),1)):round(Gold_answer2(xx));
                ind1 = [ind1,sub2ind(size(dImg),ranges,xx*ones(1,length(ranges)))];
            end
            dImg_new2(ind1) = layer_id-1;%round(median(dImg(ind1)));
            Gold_answer_prev2 = Gold_answer2;
            
        end
        
        ind1 = [];
        for xx = X_non_NAN
            ranges = round(max(Gold_answer_prev1(xx),1)):size(dImg,1);
            ind1 = [ind1,sub2ind(size(dImg),ranges,xx*ones(1,length(ranges)))];
        end
        dImg_new1(ind1) = layer_id;%median(dImg(ind1));
        dImg_new1(manualFluid1_new(:,:,im_id)~=0) = layer_id+1;
        a = round((dImg_new1(:,X_non_NAN)/max(max(dImg_new1(:,X_non_NAN))))*255);
%         subplot(1,3,2); imshow(uint8(a))
        images_grndtrth1(:,:,im_id) = dImg_new1(:,X_non_NAN);    
        temp = double(imresize(uint8(dImg_new1(:,X_non_NAN)),[image_size,image_size]));
        temp(temp>=10) = 9;
        temp(temp<=0) = 0;
        images_grndtrth1_resize(:,:,im_id) = temp;
        uu = unique(images_grndtrth1_resize(:,:,im_id));
        if any(uu<0 | uu>9)
            'errooooooooorrr'
        end      
        ind1 = [];
        for xx = X_non_NAN
            ranges = round(max(Gold_answer_prev2(xx),1)):size(dImg,1);
            ind1 = [ind1,sub2ind(size(dImg),ranges,xx*ones(1,length(ranges)))];
        end
        dImg_new2(ind1) = layer_id;%median(dImg(ind1));
        dImg_new2(manualFluid2_new(:,:,im_id)~=0) = layer_id+1;
        a = round((dImg_new2(:,X_non_NAN)/max(max(dImg_new2(:,X_non_NAN))))*255);
%         subplot(1,3,3); imshow(uint8(a))
        images_grndtrth2(:,:,im_id) = dImg_new2(:,X_non_NAN);
        temp = double(imresize(uint8(dImg_new2(:,X_non_NAN)),[image_size,image_size])); 
        temp(temp>=10) = 9;
        temp(temp<=0) = 0;
        images_grndtrth2_resize(:,:,im_id) =  temp;      
        uu = unique(images_grndtrth2_resize(:,:,im_id));
        if any(uu<0 | uu>9)
            'errroooorrr'
        end
    imshow(label2rgb(dImg_new2(:,X_non_NAN),CM),'Parent',ax1)
    set(ax1, 'box', 'on', 'Visible', 'on', 'xtick', [], 'ytick', [])
    saveas(gcf,strcat('grndtrth_data\Img_',num2str(ff),'_',num2str(im_id),'.jpg'))        
    end

    save(strcat(sprintf('Subject_%.2d_128_cropped_resize',ff),'_grndtrth.mat'),'images_grndtrth1','images_grndtrth2','images_grndtrth1_resize','images_grndtrth2_resize')
end