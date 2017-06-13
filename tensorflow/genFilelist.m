root = '../pc_real_bias_nolookat_remove/train_less/';
class = dir(root);
filelist = cell(1);
index = 1;
for i = 3:size(class,1)
    filename = dir(fullfile(root,class(i).name));    
    for j = 3:size(filename,1)
        filelist{index} = fullfile(class(i).name,filename(j).name);
        index = index + 1;
    end
end

randon = randperm(size(filelist,2));
fileID = fopen('filelist_less.txt','w');
for i = 1:size(randon,2)
    fprintf(fileID, [fullfile(filelist{randon(i)}),'\n']);
%     fprintf(fileID, '\n');
end
fclose(fileID);
