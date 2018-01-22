import numpy as np
from matplotlib import pyplot as plt
import pyfits as pf
import scipy.ndimage as snd
from hcipy import *

# This function generates one spider. The parameter angle_on_outer_diameter gives angle of the point on the outer diameter 
# where the spider starts. For the VLT this is 0,90,180 and 270 degrees at R = 8439.7 mm, and normalized R = 0.5.
# It uses the coordinates of the point where the spider is attached to the secondary mirror to calculate the angle of 
# the spider. A grid is created in this direction with zero on the outer diameter. The spider in this coordinate system is
# just everything smaller than half the spiderwidth in both directions and is cutoff somewhere behid the secondary.
def make_spider(grid, angle_on_outer_diameter, spider_offset,spider_width):
	R = 0.5
	shift_to_start_spider = (R*np.cos(angle_on_outer_diameter),R*np.sin(angle_on_outer_diameter))
	grid2 = grid.shifted(shift_to_start_spider)
	dy = shift_to_start_spider[1]-spider_offset[1]
	dx = shift_to_start_spider[0]-spider_offset[0]
	angle_of_spider = -np.arctan2(dx,dy)
	spider = (np.abs(grid2.x * np.cos(angle_of_spider) + grid2.y * np.sin(angle_of_spider))) < (spider_width/2.)
	one_side_mask =(grid.x * np.cos(angle_of_spider+np.pi/2.) + grid.y * np.sin(angle_of_spider+np.pi/2.))>0
	spider=  spider*(1-one_side_mask)
	return spider


# This function generates a box by shifting the grid to the center of the box and selecting everything smaller than the
# given dimensions.
def add_box(grid, loc, dimension):
	grid2 = grid.shifted(loc)
	box = (np.abs(grid2.x)<(dimension[0]/2))*(np.abs(grid2.y)<(dimension[1]/2))
	return box

# This function erodes (=blurring) the pupil with a kernel. The binary erosion is faster than a convolution most of the times.
def Blur_pupil(nx,pupil, pupil_grid, diameter):
	nx2 = int(nx*diameter*2+1)
	grid2 = make_pupil_grid(nx2,1)
	kernel = circular_aperture(1)(grid2)
	blurred_pupil = (snd.morphology.binary_erosion(pupil.reshape(nx,nx),structure = kernel.reshape(nx2,nx2),iterations = 1).ravel())
	return Field(blurred_pupil,pupil_grid)



#### Definitions ####



Pivot_point_spiders = 4219.7*2. #mm pivot point, secondory support structure strut crosses 0 (1)  
M1_ERIS = 8119.6 #mm Entrance pupil dia. (2)
M2_ERIS = 775*2. #DSM wind screen external diameter (3)
M2_ERIS_no_baffle = 646.5*2.

angle_between_spiders = 101 #degrees (1)
offset_spider_to_center = 404.5 #mm (1)

spider_width_VLT = 40 #mm (1) (estimated)
outer_diameter_M3_stow = 1070 #Diameter containing emission from M3 in stow position. (3)

Diameter_array = Pivot_point_spiders #physical_size of the array as projected onto M1


#Defube blurring diameters
um =1
mm =  1e3*um


radius_ERIS_pupil_ray_traced = 6.025*mm #mm half instensity point ray traced pupil (4)
scaling_ERIS_pupil_to_M1_dimensions = M1_ERIS/(radius_ERIS_pupil_ray_traced*2.)
blurring_out_of_pupil_plane = 110 *um #micron as decided by Matthew Kenworthy and David Doelman from C29 to C30 in ERIS_Pupil_alignment_tolerances_v2.xlsx
blurring_flexing_motion = 154 * um # quadratic plus half linear from Table 2 #300 micron absolute worst case, could be 150 and would be OK as decided by Matthew Kenworthy and David Doelman 12-01-2018

num_wanted_pix = 2448 #change here the number of pixels that the final array should have.
to_num_pix = 2448/num_wanted_pix #2448 = num pixels for a pixelsize of 5.*um/0.977427386, where 0.977427386 is the demagnification for out of pupil plane effects

pixelsize = 5.*um/0.977427386*to_num_pix

nx =  int(2*radius_ERIS_pupil_ray_traced*(Pivot_point_spiders/M1_ERIS)/(pixelsize))  #Array size, NOT outer diameter in pixels!

print('The size of the array is {0}x{0} pixels'.format(nx))
#print('The physical size of the array = {0} mm'.format(np.round(nx*pixelsize/mm,decimals = 3)))
print('The physical size of the array = {0} mm'.format(np.round(nx*5.0*um/mm,decimals = 3)))
print('The physical size of the ERIS entrance pupil = {0} mm'.format(np.round(M1_ERIS/Pivot_point_spiders*nx*5.0*um/mm,decimals = 3)))


#### Calculate spider coordinates ####

#Get the offset and angle of the location where the spider is attached to the secondary in polar coordinates
spider_offset_a = np.radians(90+(angle_between_spiders-90)/2.)
spider_offset_r = offset_spider_to_center/Diameter_array

#Calculate points where the spider is attached to the secondary in cartesian coordinates
spider_offset1 = (spider_offset_r*np.cos(spider_offset_a),-spider_offset_r*np.sin(spider_offset_a))
spider_offset2 = (-spider_offset_r*np.cos(spider_offset_a-np.pi/2.),-spider_offset_r*np.sin(spider_offset_a-np.pi/2.))
spider_offset3 = (spider_offset_r*np.cos(spider_offset_a-np.pi),-spider_offset_r*np.sin(spider_offset_a-np.pi))
spider_offset4 = (spider_offset_r*np.cos(spider_offset_a-np.pi/2.),-spider_offset_r*np.sin(spider_offset_a-np.pi/2.))


spider_width = spider_width_VLT/Diameter_array


#### Make pupil ####
pupil_grid = make_pupil_grid(nx,1)

aperture = circular_aperture(M1_ERIS/Diameter_array)(pupil_grid)-circular_aperture(M2_ERIS/Diameter_array)(pupil_grid)

spider1 = make_spider(pupil_grid, np.pi, spider_offset1,spider_width)
spider2 = make_spider(pupil_grid, -np.pi/2., spider_offset2,spider_width)
spider3 = make_spider(pupil_grid, 0, spider_offset3,spider_width)
spider4 = make_spider(pupil_grid, np.pi/2., spider_offset4,spider_width)

box = add_box(pupil_grid,(-outer_diameter_M3_stow/2./Diameter_array,0), (outer_diameter_M3_stow/Diameter_array,M2_ERIS_no_baffle/Diameter_array))

VLT_ERIS = aperture-spider1-spider2-spider3-spider4-box
VLT_ERIS[VLT_ERIS<0] = 0

#### Undersizing the pupil ####
Blur_kernel_oopp_projected_M1 = blurring_out_of_pupil_plane*scaling_ERIS_pupil_to_M1_dimensions

VLT_ERIS_blurred_1 = Blur_pupil(nx,VLT_ERIS,pupil_grid, Blur_kernel_oopp_projected_M1/Diameter_array)

Blur_kernel_flexing_motion_projected_M1 = blurring_flexing_motion*scaling_ERIS_pupil_to_M1_dimensions

VLT_ERIS_blurred_2 = Blur_pupil(nx,VLT_ERIS_blurred_1,pupil_grid, Blur_kernel_flexing_motion_projected_M1/Diameter_array)


FitsOut = pf.HDUList()
FitsOut.append(pf.ImageHDU(1.*VLT_ERIS.reshape((nx,nx))))
FitsOut.writeto('VLT_ERIS_pupil_A_entrance_pupil_SkyBaffle.fits.gz', clobber = True)

FitsOut = pf.HDUList()
FitsOut.append(pf.ImageHDU(1.*VLT_ERIS_blurred_1.reshape((nx,nx))))
FitsOut.writeto('VLT_ERIS_pupil_B_undersized_out_of_pupil_plane_SkyBaffle.fits.gz', clobber = True)

FitsOut = pf.HDUList()
FitsOut.append(pf.ImageHDU(1.*VLT_ERIS_blurred_2.reshape((nx,nx))))
FitsOut.writeto('VLT_ERIS_pupil_D_tightly_undersized_ZEMAX_corrected_SkyBaffle.fits.gz', clobber = True)



plt.figure()
imshow_field(VLT_ERIS,pupil_grid)
plt.title('Pupil A, entrance pupil ERIS')
#plt.savefig('VLT_ERIS_pupil_A_entrance_pupil.pdf')
plt.figure()
imshow_field(VLT_ERIS_blurred_1,pupil_grid)
plt.title('Pupil B, undersized pupil ERIS (out of pupil plane)')
#plt.savefig('VLT_ERIS_pupil_B_undersized_out_of_pupil_plane.pdf')
plt.figure()
imshow_field(VLT_ERIS_blurred_2,pupil_grid)
plt.title('Pupil D, tightly undersized pupil ERIS (all uncertainties)')
#plt.savefig('VLT_ERIS_pupil_D_tightly_undersized_all_uncertainties.pdf')
plt.show()







#### references ####
#1) VLT-SPE-AES-11310-0006, drawing no. 0-101010-0000
#2) "Table 4. Optical implementing BLF =500", VLT-TRE-ERI-14401-3103 page 8 of 14. 
#3) ERIS Memo No.: OAA-17-001, page 3/5
#4) ERIS-NIX Pupil mask requirements Specification VLT-SPE-ERI-14402-2009 page 10 of 24 Fig. 5 Cam3





