function visualizeErrorImage()
root = '/home/bonnie/Desktop/itrichess/pc_real_bias_nolookat_remove/val/';
fileID = fopen('errorList.txt','r');
index = 1;
A = textscan(fileID,'%s,%s');
% aa
fclose(fileID);
path = A{1,1};
class = A{1,2};
for i = 1:size(path)
    load(fullfile(root,cell2mat(path(i))));
    testpc(answer)
    gt = strsplit(cell2mat(path(i)),'/');
    string = strcat('GT:',gt(1),' , Guess:' , class(i));
    title(string)
    pause;
end
