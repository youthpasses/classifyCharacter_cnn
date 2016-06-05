function cor2img(filename)
	fin=fopen(filename,'r');
	fgetl(fin);
	im_nlsize=[64,64];
	re_im=[];
	i=1;
	while~feof(fin)
		dt=str2num(fgetl(fin));
		re_im(i,:)=dt;
		i=i+1;
    end
    fclose(fin);
	x_min=min(re_im(:,1))-1;
	x_max=max(re_im(:,1))+1;
	y_min=min(re_im(:,2))-1;
	y_max=max(re_im(:,2))+1;
	re_im(:,1)=round((re_im(:,1)-x_min)*(im_nlsize(1)-1)/(x_max-x_min)+1);
	re_im(:,2)=round((re_im(:,2)-y_min)*(im_nlsize(2)-1)/(y_max-y_min)+1);
    
    
	im_nlsize=[65,65];
	grey_im=zeros(im_nlsize);
	grey_im=double(grey_im);
    
    for j=1:i-1
        grey_im(re_im(j,2),re_im(j,1))=1;
    end
    grey_im=flipud(grey_im);
    imwrite(grey_im, [filename, '.png']);
% 	imshow(grey_im);
    
    
% filename = '0_1.txt'
% cor2img(filename)
%{
	set(gcf,'Color','w');%??????????????
	set(gca,'units','pixels','Visible','off');  %??????????pixel,????????????
	hold on
    plot(re_im(:,1),re_im(:,2),'color','w','LineWidth',1);
	frame=getframe(gcf);
	im=frame2im(frame);
	im=im2double(im);
	%imwrite(im,'E:\b.jpg');
%}