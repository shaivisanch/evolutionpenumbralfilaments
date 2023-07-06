pro alignedmovie_skt4Shaivi 

cd,'/sanhome/shaivi/data/Phil_Shirts/'

file='SPcubes_20170915_224806_conti_index.fits'
read_sotsp, file, index, continuum
index2map, index, continuum, map

FieldStrength = 'SPcubes_20170915_224806_magfield_index.fits'
read_sotsp, FieldStrength, index5, data5
Bb = data5

FieldInclin = 'SPcubes_20170915_224806_magincl_index.fits'
read_sotsp, FieldInclin, index6, data6
gamma = data6

Bz = Bb*cos(gamma*!dtor)

index2map, index, Bz, Bzmap

cd,'/sanhome/shaivi/data/idlpro/'


stop
end

 ;plot_image[list[130,1]],  yticks = 1, ytickname = REPLICATE(' ', 20), dmin = 0, dmax = 700, xrange = [index[list[130,0]].xcen-range,(index[list[130,0]].xcen+range+5)], yrange = [index[list[130,0]].ycen-range1+1.5,(index[list[130,0]].ycen+range1-2.5)], position =[0.18, 0.05, 0.34, 0.95],/notitle,/noerase 


