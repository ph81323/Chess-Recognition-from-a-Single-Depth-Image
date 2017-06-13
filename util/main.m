% clear
% threshold = 1;
outfileroot = '/home/bonnie/pc_bias_remove/train/';
% len = 100;
ybias = 0;
zbias = 0;
threshold_shift = 1;

path = 'chessmodel/king/model.obj';
% OBJ=read_wobj(path);
offset = 0;
% parfor ybias_t = 1:3
%     for zbias = -5:5:5
        for threshold = 0.1:0.1:1
%             for threshold_shift = 0.1:0.1:1
%                 ybias = ybias_t * 5 - 10;
%                 offset = (threshold_shift-0.1) / 0.1 * 60 + (threshold-0.1) / 0.1 * 600 + (ybias+5)/5*18000+(zbias+5)/5*6000;
                offset = (threshold - 0.1) / 0.1 * 1860;
                pcFromModel_new(outfileroot,OBJ,'king',offset,threshold,ybias,zbias,true,threshold_shift)
%                offset = offset + 1860;
%             end            
        end
%     end
% end

path = 'chessmodel/bishop/model.obj';
OBJ=read_wobj(path);
offset = 0;
% parfor ybias_t = 1:3
%     for zbias = -5:5:5
        for threshold = 0.1:0.1:1
%             for threshold_shift = 0.1:0.1:1
%                 ybias = ybias_t * 5 - 10;
%                 offset = (threshold_shift-0.1) / 0.1 * 60 + (threshold-0.1) / 0.1 * 600 + (ybias+5)/5*18000+(zbias+5)/5*6000;
                offset = (threshold - 0.1) / 0.1 * 1860;
                pcFromModel_new(outfileroot,OBJ,'bishop',offset,threshold,ybias,zbias,true,threshold_shift)

%             end  
        end
%     end
% end

% clear
path = 'chessmodel/knight/model.obj';
OBJ=read_wobj(path);
offset = 0;
% parfor ybias_t = 1:3
%     for zbias = -5:5:5
        for threshold = 0.1:0.1:1
%             for threshold_shift = 0.1:0.1:1
%                 ybias = ybias_t * 5 - 10;
%                 offset = (threshold_shift-0.1) / 0.1 * 60 + (threshold-0.1) / 0.1 * 600 + (ybias+5)/5*18000+(zbias+5)/5*6000;
                 offset = (threshold - 0.1) / 0.1 * 1860;
                pcFromModel_new(outfileroot,OBJ,'knight',offset,threshold,ybias,zbias,true,threshold_shift)
%                offset = offset + 1860;
%             end 
        end
%     end
% end

% clear
path = 'chessmodel/pawn/model.obj';
OBJ=read_wobj(path);
offset = 0;
% parfor ybias_t = 1:3
%     for zbias = -5:5:5
        for threshold = 0.1:0.1:1
%             for threshold_shift = 0.1:0.1:1
%                 ybias = ybias_t * 5 - 10;
%                 offset = (threshold_shift-0.1) / 0.1 * 60 + (threshold-0.1) / 0.1 * 600 + (ybias+5)/5*18000+(zbias+5)/5*6000;
                offset = (threshold - 0.1) / 0.1 * 1860;
                pcFromModel_new(outfileroot,OBJ,'pawn',offset,threshold,ybias,zbias,true,threshold_shift)
%                offset = offset + 1860;
%             end 
        end
%     end
% end

% clear
path = 'chessmodel/queen/model.obj';
OBJ=read_wobj(path);
offset = 0;
% parfor ybias_t = 1:3
%     for zbias = -5:5:5
        for threshold = 0.1:0.1:1
%             for threshold_shift = 0.1:0.1:1
%                 ybias = ybias_t * 5 - 10;
%                 offset = (threshold_shift-0.1) / 0.1 * 60 + (threshold-0.1) / 0.1 * 600 + (ybias+5)/5*18000+(zbias+5)/5*6000;
                offset = (threshold - 0.1) / 0.1 * 1860;
                pcFromModel_new(outfileroot,OBJ,'queen',offset,threshold,ybias,zbias,true,threshold_shift)
%                offset = offset + 1860;
%             end 
        end
%     end
% end

path = 'chessmodel/rook/model.obj';
OBJ=read_wobj(path);
offset = 0;
% parfor ybias_t = 1:3
%     for zbias = -5:5:5
        for threshold = 0.1:0.1:1
%             for threshold_shift = 0.1:0.1:1
%                 ybias = ybias_t * 5 - 10;
%                 offset = (threshold_shift-0.1) / 0.1 * 60 + (threshold-0.1) / 0.1 * 600 + (ybias+5)/5*18000+(zbias+5)/5*6000;
                 offset = (threshold - 0.1) / 0.1 * 1860;
                pcFromModel_new(outfileroot,OBJ,'rook',offset,threshold,ybias,zbias,true,threshold_shift)
%                offset = offset + 1860;
%             end 
        end
%     end
% end
