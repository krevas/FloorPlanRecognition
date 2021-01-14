import os
import time
import argparse
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import streamlit as st
from scipy.misc import imread, imsave, imresize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def main(args):
    st.title("Deep Floor Plan Recognition Using a Multi-Task Network with Room-Boundary-Guided Attention")

    image = Image.open('demo/demo.JPG')
    st.image(image, use_column_width=True)

    side_image = Image.open('demo/color.JPG')

    st.text("")
    link = '[GitHub](https://github.com/zlzeng/DeepFloorplan)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[PAPER](https://arxiv.org/abs/1908.11025)'
    st.markdown(link, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        im = imread(image_file, mode='RGB')
        im = im.astype(np.float32)
        im = imresize(im, (512,512,3)) / 255.

        #file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
        #st.write(file_details)

                        
    if st.button("Analysis", key='message'):
        with st.spinner('Wait for it...'):
            time.sleep(1)
        # load input
        if image_file is None:
            st.success('Using a sample floorplan image file.')
            im = imread(args.im_path, mode='RGB')
            im = im.astype(np.float32)
            im = imresize(im, (512,512,3)) / 255.

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # create tensorflow session
        with tf.compat.v1.Session(config=config) as sess:
            
            # initialize
            sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer()))
            with st.echo():
                # restore pretrained model
                saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d/pretrained_r3d.meta')
                saver.restore(sess, tf.train.latest_checkpoint('./pretrained/pretrained_r3d'))

                # get default graph
                graph = tf.compat.v1.get_default_graph()

                # restore inputs & outpus tensor
                x = graph.get_tensor_by_name('inputs:0')
                room_type_logit = graph.get_tensor_by_name('Cast:0')
                room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

                # infer results
                [room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
                                                feed_dict={x:im.reshape(1,512,512,3)})
                room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

                # merge results
                floorplan = room_type.copy()
                floorplan[room_boundary==1] = 9
                floorplan[room_boundary==2] = 10
                floorplan_rgb = ind2rgb(floorplan)

                # plot results
                plt.subplot(121)
                plt.imshow(im)
                plt.title('Input')
                plt.subplot(122)
                plt.imshow(floorplan_rgb/255.)
                plt.title('Output')
                st.pyplot()
                st.sidebar.image(side_image, use_column_width=True)
                #plt.show()

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
