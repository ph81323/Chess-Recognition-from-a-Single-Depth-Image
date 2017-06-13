local matio = require 'matio'
path = '/home/bonnie/Desktop/itrichess/itrichess_real/train/king/result_9300.mat'
local tmp =  matio.load(path, 'answer')
local size = tmp:size()
--local temp = {}
print("x = "..tostring(size[1]))
print("y = "..tostring(size[2]))
print("z = "..tostring(size[3]))
local temp = tmp[{{51,100},{},{}}]
--for i = 51,size[1] do
--	temp[i-50] = {}
--	for j = 1,size[2] do
--		temp[i-50][j] = {}
--		for k = 1,size[3] do
--			temp[i-50][j][k] = tmp[i][j][k]
--		end
--	end
--end
--print(temp)
matio.save('test.mat',temp);
