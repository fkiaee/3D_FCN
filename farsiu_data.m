clc
clear all
close all
%%%% we can make sure that manula annotations are avaiable from 86-667 in second dimention of 768
%%%% we can make sure that images are not highly white from 94-419 in first dimention of 496
% cd('D:\HAMEDFAHIMI\LiveWire\OcularData\2015_BOE_Chiu')
cd('F:\Hamed\postdoc\livewire\OcularData\2015_BOE_Chiu')
num_individual = 10;
d1_min = 0; d2_min = 0; %d1-377-493
d1_max = 600; d2_max = 800;
h1 = figure; h2 = figure;
xrange = 135:600;
yrange = 94:313;%419;
removed_size = 93; 
size2 = length(xrange); size1 = length(yrange);%135:634;
image_size = 128;
for ff = 1:num_individual
    %     sprintf('%.2d',1)
    load(sprintf('Subject_%.2d.mat',ff))
    images_new = zeros(size1,size2,11);
    images_resize = zeros(image_size,image_size,11);
    manualLayers2_new = zeros(8,size2,11);
    manualLayers1_new = zeros(8,size2,11);
    manualFluid1_new = zeros(size1,size2,11);
    manualFluid2_new = zeros(size1,size2,11);
    avails = 1:size(images,3);
    avails = avails(~(all(squeeze(all(isnan(manualLayers1))))));
    images = images(:,:,avails);
    for im_id = 1:size(images,3)
        %         MLS1 = manualLayers1(:,~(all(squeeze(all(isnan(manualLayers1)))')),avails(im_id));
        %         MLS2 = manualLayers2(:,~(all(squeeze(all(isnan(manualLayers2)))')),avails(im_id));
        MLS1 = manualLayers1(:,:,avails(im_id));
        MLS2 = manualLayers2(:,:,avails(im_id));
        % MFD = automaticFluidDME(:,any(squeeze(any(automaticFluidDME))'),any(squeeze(any(automaticFluidDME))));
        %         MFD1 = manualFluid1(:,any(squeeze(any(manualFluid1))'),any(squeeze(any(manualFluid1))));
        %         MFD2 = manualFluid2(:,any(squeeze(any(manualFluid2))'),any(squeeze(any(manualFluid2))));
        Img = images(:,:,im_id);
        removed_rows = ones(1,496); removed_rows(yrange) = 0;                  %%%%(sum((Img==255)')>5  or all((Img==255)')
        %         removed_size = sum(removed_rows);
        MLS1 = MLS1 - removed_size;
        MLS2 = MLS2 - removed_size;
        MLS1 = MLS1(:,xrange,:);        MLS2 = MLS2(:,xrange,:);
        MFD1 = manualFluid1(~removed_rows, xrange ,avails(im_id));
        MFD2 = manualFluid2(~removed_rows,xrange,avails(im_id));
        Img = Img(~removed_rows,xrange);
        %                 val_min1 = 800; val_max1 = 0;
        %                 for jj = 1:8
        %                     d1 = find(~isnan(MLS1(jj,:)));
        %                     if min(d1)<val_min1
        %                         val_min1 = min(d1);
        %                     end
        %                     if max(d1)>val_max1
        %                         val_max1 = max(d1);
        %                     end
        %                 end
        %                 val_min2 = 800; val_max2 = 0;
        %                 for jj = 1:8
        %                     d1 = find(~isnan(MLS2(jj,:)));
        %                     if min(d1)<val_min2
        %                         val_min2 = min(d1);
        %                     end
        %                     if max(d1)>val_max2
        %                         val_max2 = max(d1);
        %                     end
        %                 end
        %                 %minmax and maxmin
        %                 val_min = min(val_min1,val_min2); val_max = max(val_max1,val_max2);
        %                 if d1_min<val_min; d1_min = val_min; end
        %                 if d1_max>val_max; d1_max = val_max; end
        %
        %                 d2 = find(~(sum((Img==255)')>5));
        %                 im_min = min(d2); im_max = max(d2);
        %                 if d2_min<im_min; d2_min = im_min; end
        %                 if d2_max>im_max; d2_max = im_max; end        
        images_new(:,:,im_id) =  Img;
        images_resize(:,:,im_id) =  double(imresize(uint8(Img),[image_size,image_size]));%Img;
        uu = images_resize(:,:,im_id);
        uu = uu(:);
        if any(uu<0 | uu>255)
            error('errroooorrr')
            uu'
            ff
        end        
        %imshow(images_resize(:,:,im_id))
        manualLayers1_new(:,:,im_id) = MLS1;
        manualLayers2_new(:,:,im_id) = MLS2;
        manualFluid1_new(:,:,im_id) = MFD1;
        manualFluid2_new(:,:,im_id) = MFD2;
        Img1 = Img;
        Img1(MFD1(:,:,1)~=0)=255;
        Img2 = Img;
        Img2(MFD2(:,:,1)~=0)=255;
        %                 if any(any(isnan(MLS1))); error('aaa'); end
        %                         if any(any(isnan(MLS2))); error('aaa'); end
        %         figure(h1)
        %         imshow(uint8(Img1))
        %         hold on; plot(MLS1')
        %         saveas(gcf,strcat('images\Img_',num2str(ff),'_',num2str(im_id),'_1.jpg'))
        %
        %         figure(h2)
        %         imshow(uint8(Img2))
        %         hold on; plot(MLS2')
        %         saveas(gcf,strcat('images\Img_',num2str(ff),'_',num2str(im_id),'_2.jpg'))
    end   
    save(strcat(sprintf('Subject_%.2d',ff),'_128_cropped_resize.mat'),'images_new','images_resize','manualLayers1_new','manualLayers2_new','manualFluid1_new','manualFluid2_new')
end