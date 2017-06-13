% threshold = 1;
outfileroot = '/home/bonnie/many_nolookat_remove/train/';
% ybias = 0;
% zbias = 0;
modellist = dir('chessmodel/king');
% path = 'chessmodel/king/model.obj';
% OBJ=read_wobj(path);
offset = 0;
parfor i = 3:size(modellist,1)-1
    path = ['chessmodel/king/',modellist(i).name,'/model.obj'];
    OBJ=read_wobj(path);
    for ybias_t = 1:3
        for zbias = -5:5:5
            for threshold = 0.1:0.1:1
                ybias = ybias_t * 5 - 10;
                offset = threshold / 0.1 * 60 + (ybias+5)/5*1800+(zbias+5)/5*600+(i-3)*5400;
                pcFromModel_new(outfileroot,OBJ,'king',offset,threshold,ybias,zbias,false)
                %             offset = offset + 1860;
            end
        end
    end
end

modellist = dir('chessmodel/bishop');
% path = 'chessmodel/bishop/model.obj';
% OBJ=read_wobj(path);
offset = 0;
parfor i = 3:size(modellist,1)-1
    path = ['chessmodel/bishop/',modellist(i).name,'/model.obj'];
    OBJ=read_wobj(path);
    for ybias_t = 1:3
        for zbias = -5:5:5
            for threshold = 0.1:0.1:1
                ybias = ybias_t * 5 - 10;
                offset = threshold / 0.1 * 60 + (ybias+5)/5*1800+(zbias+5)/5*600+(i-3)*5400;
                pcFromModel_new(outfileroot,OBJ,'bishop',offset,threshold,ybias,zbias,false)
                %             offset = offset + 1860;
            end
        end
    end
end

modellist = dir('chessmodel/knight');
% path = 'chessmodel/knight/model.obj';
% OBJ=read_wobj(path);
offset = 0;
parfor i = 3:size(modellist,1)-1
    path = ['chessmodel/knight/',modellist(i).name,'/model.obj'];
    OBJ=read_wobj(path);
    for ybias_t = 1:3
        for zbias = -5:5:5
            for threshold = 0.1:0.1:1
                ybias = ybias_t * 5 - 10;
                offset = threshold / 0.1 * 60 + (ybias+5)/5*1800+(zbias+5)/5*600+(i-3)*5400;
                pcFromModel_new(outfileroot,OBJ,'knight',offset,threshold,ybias,zbias,false)
                %             offset = offset + 1860;
            end
        end
    end
end

modellist = dir('chessmodel/pawn');
% path = 'chessmodel/pawn/model.obj';
% OBJ=read_wobj(path);
offset = 0;
parfor i = 3:size(modellist,1)-1
    path = ['chessmodel/pawn/',modellist(i).name,'/model.obj'];
    OBJ=read_wobj(path);
    for ybias_t = 1:3
        for zbias = -5:5:5
            for threshold = 0.1:0.1:1
                ybias = ybias_t * 5 - 10;
                offset = threshold / 0.1 * 60 + (ybias+5)/5*1800+(zbias+5)/5*600+(i-3)*5400;
                pcFromModel_new(outfileroot,OBJ,'pawn',offset,threshold,ybias,zbias,false)
                %             offset = offset + 1860;
            end
        end
    end
end

modellist = dir('chessmodel/queen');
% path = 'chessmodel/queen/model.obj';
% OBJ=read_wobj(path);
offset = 0;
parfor i = 3:size(modellist,1)-1
    path = ['chessmodel/queen/',modellist(i).name,'/model.obj'];
    OBJ=read_wobj(path);
    for ybias_t = 1:3
        for zbias = -5:5:5
            for threshold = 0.1:0.1:1
                ybias = ybias_t * 5 - 10;
                offset = threshold / 0.1 * 60 + (ybias+5)/5*1800+(zbias+5)/5*600+(i-3)*5400;
                pcFromModel_new(outfileroot,OBJ,'queen',offset,threshold,ybias,zbias,false)
                %             offset = offset + 1860;
            end
        end
    end
end

modellist = dir('chessmodel/rook');
% path = 'chessmodel/rook/model.obj';
% OBJ=read_wobj(path);
offset = 0;
parfor i = 3:size(modellist,1)-1
    path = ['chessmodel/rook/',modellist(i).name,'/model.obj'];
    OBJ=read_wobj(path);
    for ybias_t = 1:3
        for zbias = -5:5:5
            for threshold = 0.1:0.1:1
                ybias = ybias_t * 5 - 10;
                offset = threshold / 0.1 * 60 + (ybias+5)/5*1800+(zbias+5)/5*600+(i-3)*5400;
                pcFromModel_new(outfileroot,OBJ,'rook',offset,threshold,ybias,zbias,false)
                %             offset = offset + 1860;
            end
        end
    end
end
