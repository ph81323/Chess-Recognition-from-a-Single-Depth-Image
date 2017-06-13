function pcFromModel_new(outfileroot,OBJ,category,offset,threshold,ybias,zbias,lookat,threshold_shift)
% path = 'D:\(TOP2)Lab\project\ITRI\chessmodel\bishop\f1\model.obj';
% OBJ=read_wobj(path);
vertex = OBJ.vertices;
normals = OBJ.vertices_normal;
R = rotz(270);          %rotate the chess to yz plane

% shownormal(vertex,normals)

vertex = R*vertex';
vertex = vertex';
normals = R*normals';
normals = normals';


% shownormal(vertex,normals)

index = 1;
parameters = zeros(10,2);
for rotation = 0:6:354
    for height = 0:2:60
        parameters(index,:) = [rotation,height];
        index = index + 1;
    end
end

parfor index = 1:size(parameters,1)
    vertex_temp = vertex;
    normal_temp = normals;
    
    R = rotx(parameters(index,1));
    vertex_temp = R*vertex_temp';
    vertex_temp = vertex_temp';
    
    normal_temp = R*normal_temp';
    normal_temp = normal_temp';
    
    if lookat == true
        vertex_new = rotatemodel(parameters(index,1),parameters(index,2),vertex_temp,normal_temp);
    elseif lookat == false
        vertex_new = vertex_temp;
    end
    
    %translate to origin
    vertex_new(:,1) = vertex_new(:,1) - min(vertex_new(:,1))+1;
    vertex_new(:,2) = vertex_new(:,2) - min(vertex_new(:,2))+1;
    vertex_new(:,3) = vertex_new(:,3) - min(vertex_new(:,3))+1;
    
    ind = find(vertex_new(:,1) < 40);
    [xc,yc,~,~] = circlefit(vertex_new(ind,2)',vertex_new(ind,3)');
    
    %     [xc,yc,R,~] = circlefit(vertex_new(:,2)',vertex_new(:,3)');
    
    %     vertex = scale(vertex,R);
    sy = round(xc) - 25;
    sz = round(yc) - 25;
    vertex_new = round(vertex_new);
    
    x = vertex_new(:,1);
    y = vertex_new(:,2);
    z = vertex_new(:,3);
    
    y = y - sy;
    z = z - sz;
    
    y = y - ybias;
    z = z - zbias;
    
    
    %         A(1,:) = 0;
    %         testpc(A')
    
    answer = zeros(100,50,50);
    for i = 1:size(x,1)
        if x(i) > 0 && x(i) < 100 && y(i) > 0 && y(i) < 50 && z(i) > 0 && z(i) < 50
            check = rand();     %check if this pixel will shift
            shift = 0;
            if check > threshold
                continue;
                %shift = 1;%round(rand()*10-5); %shift:-5~5
            else
		check = rand();
		if check > threshold_shift
			shift = 1;
		end               
                x(i) = x(i)+shift*round(rand()*10-5);
                x(i) = max(1,x(i));
                x(i) = min(x(i),size(answer,1));
                
                y(i) = y(i)+shift*round(rand()*10-5);
                y(i) = max(1,y(i));
                y(i) = min(y(i),size(answer,2));
                
                z(i) = z(i)+shift*round(rand()*10-5);
                z(i) = max(1,z(i));
                z(i) = min(z(i),size(answer,3));
                
                answer(x(i),y(i),z(i)) = 1;
            end
            
        end
    end
    %     testpc(answer)
    %     parameters(index,2)
    filename = [outfileroot,category,'/result_',num2str(index+offset)];
    parsave(filename,answer)
    %     save(filename,'answer','-v7.3')
    
end

function parsave(filename, answer)
save(filename, 'answer','-v7.3')

function vertex_new = rotatemodel(xangle,yangle,vertex,normals)
frontvector = [0 0 -1];
meanx = (max(vertex(:,1)) + min(vertex(:,1)))/2;
meany = (max(vertex(:,2)) + min(vertex(:,2)))/2;
meanz = (max(vertex(:,3)) + min(vertex(:,3)))/2;

% R = rotx(xangle)*roty(-yangle);
R = roty(-yangle);
frontvector = R*frontvector';
index = 1;
for i = 1:size(normals,1)
    c_normal(1,1) = meanx - vertex(i,1);
    c_normal(1,2) = meany - vertex(i,2);
    c_normal(1,3) = meanz - vertex(i,3);
    c_normal = -c_normal;
    
    CosTheta = dot(frontvector,c_normal)/(norm(frontvector)*norm(c_normal));
    c_ThetaInDegrees = acosd(CosTheta);
    
    normal(1,1) = normals(i,1);
    normal(1,2) = normals(i,2);
    normal(1,3) = normals(i,3);
    %     normal = normal;
    
    CosTheta = dot(frontvector,normal)/(norm(frontvector)*norm(normal));
    ThetaInDegrees = acosd(CosTheta);
    
    %you can see this point
    if abs(ThetaInDegrees) < 90
        vertex_new(index,:) = vertex(i,:);
        index = index + 1;
    end
end
% index

function shownormal(vertex,normals)
testpc(vertex);
for i  = 1:50:size(vertex,1)
    line1 = [vertex(i,1),vertex(i,2),vertex(i,3);vertex(i,1)+normals(i,1)*2,vertex(i,2)+normals(i,2)*2,vertex(i,3)+normals(i,3)*2];
    hold on;
    line(line1(:,1),line1(:,2),line1(:,3),'Color','r')
end

function showplane()
x = meanx-10:.1:meanx+10;
y = meany-10:.1:meany+10;
y = y';
a = frontvector(1);
b = frontvector(2);
c = frontvector(3);
d = +90;
[X,Y] = meshgrid(x,y);
Z=(d- a * X - b * Y)/c;

surf(X,Y,Z)
shading flat
xlabel('x'); ylabel('y'); zlabel('z')

function vertex_new = scale(vertex,R)
standardR = 11;

x = vertex(:,1);
y = vertex(:,2);
z = vertex(:,3);
scale = standardR/R;
xt = (min(x) + max(x))/2;
yt = (min(y) + max(y))/2;
zt = (min(z) + max(z))/2;
x = (x - xt)*scale + xt;
y = (y - yt)*scale + yt;
z = (z - zt)*scale + zt;
vertex_new(:,1) = x;
vertex_new(:,2) = y;
vertex_new(:,3) = z;
