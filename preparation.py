import tensorflow as tf

def prep_network(secret):
    with tf.variable_scope('prep_net'):

        # concat_input = tf.concat([cover, secret],  axis = 4, name='concat')
        # #print(concat_input)

        conv1 = tf.layers.conv3d(secret,32,3,padding='same',name="1",activation=tf.nn.relu,data_format='channels_last')  
        conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="2",activation=tf.nn.relu,data_format='channels_last') 
        conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="3",activation=tf.nn.relu,data_format='channels_last')
        conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="4",activation=tf.nn.relu,data_format='channels_last') 
        conv1 = tf.layers.conv3d(conv1,32,3,padding='same',name="5",activation=tf.nn.relu,data_format='channels_last')
        #print(conv1)

        #v1 = tf.layers.max_pooling3d(conv1, (2,1,1), strides=(2,1,1), data_format='channels_last')
        v1 = tf.layers.conv3d(conv1,32,3,padding='same',name = '6', activation=tf.nn.relu,data_format='channels_last')
        #print(v1)
        
       
        #maxpool1 = tf.layers.max_pooling3d(conv1, 2, strides=2, data_format='channels_last')

        #print(maxpool1)
        
        conv2 = tf.layers.conv3d(v1,64,3,padding='same',name="49",activation=tf.nn.relu,data_format='channels_last')  
        conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="7",activation=tf.nn.relu,data_format='channels_last') 
        conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="8",activation=tf.nn.relu,data_format='channels_last')
        conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="9",activation=tf.nn.relu,data_format='channels_last') 
        conv2 = tf.layers.conv3d(conv2,64,3,padding='same',name="10",activation=tf.nn.relu,data_format='channels_last')
        #print(conv2)

        v2 = tf.layers.conv3d(conv2,64,3,padding='same',name = '11', activation=tf.nn.relu,data_format='channels_last')
        
       
        maxpool2 = tf.layers.max_pooling3d(conv2, 2, strides=2, data_format='channels_last')
        #print(maxpool2)
        
        conv3 = tf.layers.conv3d(maxpool2,128,3,padding='same',name="12",activation=tf.nn.relu,data_format='channels_last')  
        conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="13",activation=tf.nn.relu,data_format='channels_last') 
        conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="14",activation=tf.nn.relu,data_format='channels_last')
        conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="15",activation=tf.nn.relu,data_format='channels_last') 
        conv3 = tf.layers.conv3d(conv3,128,3,padding='same',name="16",activation=tf.nn.relu,data_format='channels_last')
        #print(conv3)

        v3 = tf.layers.conv3d(conv3,128,3,padding='same',name = '17', activation=tf.nn.relu,data_format='channels_last')
        
       
        maxpool3 = tf.layers.max_pooling3d(conv3, (1,2,2), strides=(1,2,2), data_format='channels_last')
        #print(maxpool3)
        
        conv4 = tf.layers.conv3d(maxpool3,256,3,padding='same',name="18",activation=tf.nn.relu,data_format='channels_last')  
        conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="19",activation=tf.nn.relu,data_format='channels_last') 
        conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="20",activation=tf.nn.relu,data_format='channels_last')
        conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="21",activation=tf.nn.relu,data_format='channels_last') 
        conv4 = tf.layers.conv3d(conv4,256,3,padding='same',name="22",activation=tf.nn.relu,data_format='channels_last')
        #print(conv4)

        v4 = tf.layers.conv3d(conv4,256,3,padding='same',name = '23', activation=tf.nn.relu,data_format='channels_last')
        
       
        maxpool4 = tf.layers.max_pooling3d(conv4, (1,2,2), strides=(1,2,2), data_format='channels_last')
        #print(maxpool4)
        conv5 = tf.layers.conv3d(maxpool4,512,3,padding='same',name="24",activation=tf.nn.relu,data_format='channels_last')  
        conv5 = tf.layers.conv3d(conv5,512,3,padding='same',name="25",activation=tf.nn.relu,data_format='channels_last') 
        conv5 = tf.layers.conv3d(conv5,512,3,padding='same',name="26",activation=tf.nn.relu,data_format='channels_last')

        conv5 = tf.layers.conv3d(conv5,512,3,padding='same',name="27",activation=tf.nn.relu,data_format='channels_last') 
        conv5 = tf.layers.conv3d(conv5,512,3,padding='same',name="28",activation=tf.nn.relu,data_format='channels_last')
        #print(conv5)
        upsample1 = tf.keras.layers.UpSampling3D(size=(1,2,2),  data_format='channels_last')(conv5)

       # upsample1 = conv3dTranspose(filters=256,kernel_size=3, padding='same', activation=tf.nn.relu,data_format='channels_last', data_format='channels_last')(upsample1)
        #print(upsample1)
        # #print(v4)
        concat1 = tf.concat([v4, upsample1], axis = 4, name = 'concat1')
        
        conv6 = tf.layers.conv3d(concat1,256,3,padding='same',name="29",activation=tf.nn.relu,data_format='channels_last')  
        conv6 = tf.layers.conv3d(conv6,256,3,padding='same',name="30",activation=tf.nn.relu,data_format='channels_last') 
        conv6 = tf.layers.conv3d(conv6,256,3,padding='same',name="31",activation=tf.nn.relu,data_format='channels_last')
        conv6 = tf.layers.conv3d(conv6,256,3,padding='same',name="32",activation=tf.nn.relu,data_format='channels_last') 
        conv6 = tf.layers.conv3d(conv6,256,3,padding='same',name="33",activation=tf.nn.relu,data_format='channels_last')
        #print(conv6)

        upsample2 = tf.keras.layers.UpSampling3D(size=(1,2,2),  data_format='channels_last')(conv6)
        #print(upsample2)

        concat2 = tf.concat([v3, upsample2], axis = 4, name = 'concat2')
        
        conv7 = tf.layers.conv3d(concat2,128,3,padding='same',name="34",activation=tf.nn.relu,data_format='channels_last')  
        conv7 = tf.layers.conv3d(conv7,128,3,padding='same',name="35",activation=tf.nn.relu,data_format='channels_last') 
        conv7 = tf.layers.conv3d(conv7, 128,3,padding='same',name="36",activation=tf.nn.relu,data_format='channels_last')
        conv7 = tf.layers.conv3d(conv7,128,3,padding='same',name="37",activation=tf.nn.relu,data_format='channels_last') 
        conv7 = tf.layers.conv3d(conv7, 128,3,padding='same',name="38",activation=tf.nn.relu,data_format='channels_last')
        #print(conv7)

        upsample3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2),  data_format='channels_last')(conv7)
        #print(upsample3)
        
        concat3 = tf.concat([v2, upsample3], axis = 4, name = 'concat3')
        
        conv8 = tf.layers.conv3d(concat3,64,3,padding='same',name="39",activation=tf.nn.relu,data_format='channels_last')  
        conv8 = tf.layers.conv3d(conv8,64,3,padding='same',name="40",activation=tf.nn.relu,data_format='channels_last') 
        conv8 = tf.layers.conv3d(conv8,64,3,padding='same',name="41",activation=tf.nn.relu,data_format='channels_last')
        conv8 = tf.layers.conv3d(conv8,64,3,padding='same',name="42",activation=tf.nn.relu,data_format='channels_last') 
        conv8 = tf.layers.conv3d(conv8,64,3,padding='same',name="43",activation=tf.nn.relu,data_format='channels_last')
        #print(conv8)
        
        upsample4 = tf.keras.layers.UpSampling3D(size=(1,2,2),  data_format='channels_last')(conv8)
        #print(upsample4)
        
        #concat4 = tf.concat([v1, upsample4], axis = 4, name = 'concat4')
        
        conv9 = tf.layers.conv3d(upsample4,32,3,padding='same',name="44",activation=tf.nn.relu,data_format='channels_last')  
        conv9 = tf.layers.conv3d(conv9,32,3,padding='same',name="45",activation=tf.nn.relu,data_format='channels_last')
        conv9 = tf.layers.conv3d(conv9,32,3,padding='same',name="46",activation=tf.nn.relu,data_format='channels_last')  
        conv9 = tf.layers.conv3d(conv9,32,3,padding='same',name="47",activation=tf.nn.relu,data_format='channels_last')  
        output = tf.layers.conv3d(conv9,3,3,padding='same',name="48",activation=tf.nn.sigmoid)
        #print("\n\n\revealing_network shape : ",output,"\n\n\n")
        ##print("Hiding_Net")
        return output

# input_shape=(None,8,160,120,3)
# container = tf.placeholder(shape=input_shape, dtype=tf.float32, name='prep_input')
# prep_network(container)
    

    
    
    
    
    
    
    
    

    
    