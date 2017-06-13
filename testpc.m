function testpc(pc)
%This is a general function for shwowing 3D pointclouds.
%Input format can be Nx3 or NxMxK
%This is written by YuAn (Dec 2016)
if size(pc,3)~= 1
    size_x = size(pc,1);
    size_y = size(pc,2);
    size_z = size(pc,3);
    figure;
    [r,c,v] = ind2sub(size(pc),find(pc==1));
    plot3(r(:),c(:),v(:),'.','Color','b');
    hold on;

else
    x = pc(:,1);
    y = pc(:,2);
    z = pc(:,3);
    figure;
    scatter3(x,y,z,'.');
end
view(3), axis vis3d, box on, rotate3d on
xlabel('x'), ylabel('y'), zlabel('z')
% xlim([0,100])
% ylim([0,50])
% zlim([0,50])
end
