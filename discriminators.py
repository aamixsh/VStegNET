import tensorflow as tf


def discriminator_loss(real, fake):
    # real_loss = 0
    # fake_loss = 0

    # if loss_func.__contains__('wgan') :
    #     real_loss = -tf.reduce_mean(real)
    #     fake_loss = tf.reduce_mean(fake)

    # if loss_func == 'lsgan' :
    #     real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
    #     fake_loss = tf.reduce_mean(tf.square(fake))

    # if loss_func == 'gan' or loss_func == 'dragan' :
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    # if loss_func == 'hinge' :
    #     real_loss = tf.reduce_mean(relu(1.0 - real))
    #     fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


def disc_container_cover_network(tensor, reuse=False):
	with tf.variable_scope("disc_con_cov", reuse=reuse):

		conv1 = tf.layers.conv3d(tensor,32,3,padding='same',name="1",activation=tf.nn.relu,data_format='channels_last')  
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="2",activation=tf.nn.relu,data_format='channels_last') 
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="3",activation=tf.nn.relu,data_format='channels_last')
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="4",activation=tf.nn.relu,data_format='channels_last')  
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="5",activation=tf.nn.relu,data_format='channels_last') 

		v1 = tf.layers.max_pooling3d(conv1, (2,1,1), strides=(2,1,1), data_format='channels_last')
		v1 = tf.layers.conv3d(v1,32,3,padding='valid',name = '6', activation=tf.nn.relu,data_format='channels_last')
		
		maxpool1 = tf.layers.max_pooling3d(conv1, 2, strides=2, data_format='channels_last')

		conv2 = tf.layers.conv3d(maxpool1,64,3,padding='same',name="7",activation=tf.nn.relu,data_format='channels_last')  
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="8",activation=tf.nn.relu,data_format='channels_last') 
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="9",activation=tf.nn.relu,data_format='channels_last')
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="10",activation=tf.nn.relu,data_format='channels_last') 
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="11",activation=tf.nn.relu,data_format='channels_last')

		v2 = tf.layers.conv3d(conv2,64,3,padding='valid',name = '12', activation=tf.nn.relu,data_format='channels_last')
		
		maxpool2 = tf.layers.max_pooling3d(conv2, 2, strides=2, data_format='channels_last')
		
		conv3 = tf.layers.conv3d(maxpool2,128,3,padding='same',name="13",activation=tf.nn.relu,data_format='channels_last')  
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="14",activation=tf.nn.relu,data_format='channels_last') 
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="15",activation=tf.nn.relu,data_format='channels_last')
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="16",activation=tf.nn.relu,data_format='channels_last') 
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="17",activation=tf.nn.relu,data_format='channels_last')

		v3 = tf.layers.conv3d(conv3,128,2,padding='valid',name = '18', activation=tf.nn.relu,data_format='channels_last')
		
		maxpool3 = tf.layers.max_pooling3d(conv3, (1,2,2), strides=(1,2,2), data_format='channels_last')
		
		conv4 = tf.layers.conv3d(maxpool3,256,3,padding='same',name="19",activation=tf.nn.relu,data_format='channels_last')  
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="20",activation=tf.nn.relu,data_format='channels_last') 
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="21",activation=tf.nn.relu,data_format='channels_last')
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="22",activation=tf.nn.relu,data_format='channels_last') 
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="23",activation=tf.nn.relu,data_format='channels_last')

		v4 = tf.layers.conv3d(conv4,256,2,padding='same',name = '24', activation=tf.nn.relu,data_format='channels_last')

		ret = tf.layers.conv3d(conv4, 16, 2, name='25', padding='valid')
		ret = tf.reshape(ret, (-1, 29, 39, 16))
		ret = tf.layers.conv2d(ret, 1, 29, name='26')

		return ret


def disc_secret_revealed_network(tensor, reuse=False):
	with tf.variable_scope("disc_sec_rev", reuse=reuse):

		conv1 = tf.layers.conv3d(tensor,32,3,padding='same',name="1",activation=tf.nn.relu,data_format='channels_last')  
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="2",activation=tf.nn.relu,data_format='channels_last') 
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="3",activation=tf.nn.relu,data_format='channels_last')
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="4",activation=tf.nn.relu,data_format='channels_last')  
		conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="5",activation=tf.nn.relu,data_format='channels_last') 

		v1 = tf.layers.max_pooling3d(conv1, (2,1,1), strides=(2,1,1), data_format='channels_last')
		v1 = tf.layers.conv3d(v1,32,3,padding='valid',name = '6', activation=tf.nn.relu,data_format='channels_last')
		
		maxpool1 = tf.layers.max_pooling3d(conv1, 2, strides=2, data_format='channels_last')

		conv2 = tf.layers.conv3d(maxpool1,64,3,padding='same',name="7",activation=tf.nn.relu,data_format='channels_last')  
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="8",activation=tf.nn.relu,data_format='channels_last') 
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="9",activation=tf.nn.relu,data_format='channels_last')
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="10",activation=tf.nn.relu,data_format='channels_last') 
		conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="11",activation=tf.nn.relu,data_format='channels_last')

		v2 = tf.layers.conv3d(conv2,64,3,padding='valid',name = '12', activation=tf.nn.relu,data_format='channels_last')
		
		maxpool2 = tf.layers.max_pooling3d(conv2, 2, strides=2, data_format='channels_last')
		
		conv3 = tf.layers.conv3d(maxpool2,128,3,padding='same',name="13",activation=tf.nn.relu,data_format='channels_last')  
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="14",activation=tf.nn.relu,data_format='channels_last') 
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="15",activation=tf.nn.relu,data_format='channels_last')
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="16",activation=tf.nn.relu,data_format='channels_last') 
		conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="17",activation=tf.nn.relu,data_format='channels_last')

		v3 = tf.layers.conv3d(conv3,128,2,padding='valid',name = '18', activation=tf.nn.relu,data_format='channels_last')
		
		maxpool3 = tf.layers.max_pooling3d(conv3, (1 ,2,2), strides=(1,2,2), data_format='channels_last')
		
		conv4 = tf.layers.conv3d(maxpool3,256,3,padding='same',name="19",activation=tf.nn.relu,data_format='channels_last')  
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="20",activation=tf.nn.relu,data_format='channels_last') 
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="21",activation=tf.nn.relu,data_format='channels_last')
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="22",activation=tf.nn.relu,data_format='channels_last') 
		conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="23",activation=tf.nn.relu,data_format='channels_last')

		v4 = tf.layers.conv3d(conv4,256,2,padding='same',name = '24', activation=tf.nn.relu,data_format='channels_last')

		ret = tf.layers.conv3d(conv4, 16, 2, name='25', padding='valid')
		ret = tf.reshape(ret, (-1, 29, 39, 16))
		ret = tf.layers.conv2d(ret, 1, 29, name='26')

		return ret

# input_shape=(None,8,320,240,3)
# t = tf.placeholder(shape=input_shape, dtype=tf.float32, name='cover_input')
# disc_container_cover_network(t)