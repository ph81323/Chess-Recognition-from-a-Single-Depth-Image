for el = -89 : .5 : -70
    az = 90;
%     el = -90;
    view(az, el)
    drawnow
    frame = getframe(1);
      im = frame2im(frame);
      [imind,cm] = rgb2ind(im,256);
%       if el == -89;
%           imwrite(imind,cm,'knight_itri.gif','gif','DelayTime',0, 'Loopcount',inf);
%       else
%           imwrite(imind,cm,'knight_itri.gif','gif','DelayTime',0,'WriteMode','append');
%       end
end 
