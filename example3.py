#!/usr/bin/env python
#coding:utf-8
"""
  Author: YeliangLi
  Created: 2018/5/6
"""

import tensorflow as tf

class CustomCell(tf.contrib.rnn.RNNCell):
    def __init__(self,num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        return (self._num_units,self._num_units)

    @property
    def output_size(self):
        return self._num_units


    def zero_state(self,batch_size,dtype):
        c = tf.zeros([batch_size,self._num_units],dtype)
        return c,c

    def __call__(self,inputs,state):
        outputs = inputs + state[0]
        next_state = (outputs,outputs)
        return outputs,next_state


def custom_rnn(cell,inputs,sequence_length):
    """
    inputs:(batch_size,time_steps,dim)
    sequence_length:(batch_size,)
    """

    inputs = tf.transpose(inputs,[1,0,2]) #(time_steps,batch_size,dim)
    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output == None:
            next_cell_state = cell.zero_state(tf.shape(inputs)[1],tf.float32)
            next_loop_state = tf.ones([tf.shape(inputs)[1],1],tf.int32) * time
        else:
            next_cell_state = cell_state 
            next_loop_state = loop_state + tf.ones_like(loop_state)*time
           
        elements_finished = tf.greater_equal(time,sequence_length) 
        finished = tf.reduce_all(elements_finished)
        next_input = tf.cond(
            finished,
          lambda: tf.zeros([tf.shape(inputs)[1], tf.shape(inputs)[2]], dtype=tf.float32),
          lambda: inputs[time,:,:]) 
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)        
    emit_ta, final_cell_state, final_loop_state = tf.nn.raw_rnn(cell,loop_fn)
    outputs = tf.transpose(emit_ta.stack(),[1,0,2])
    return outputs,final_cell_state,final_loop_state


if __name__ == "__main__":
    cell = CustomCell(1)
    inputs = tf.constant([1.,2.,3.,4.,5.,6.],tf.float32,[2,3,1])
    seq_len = tf.constant([2,3],tf.int32)
    outputs,final_cell_state,final_loop_state = custom_rnn(cell,inputs,seq_len)
    
    with tf.Session() as sess:
        o1,o2,o3,o4 = sess.run([outputs,final_cell_state[0],final_cell_state[1],final_loop_state])
        print(o1)
        print(o2)
        print(o3)
        print(o4)

'''
Êä³ö½á¹û£º

[[[ 1.]
  [ 3.]
  [ 0.]]

 [[ 4.]
  [ 9.]
  [15.]]]
[[ 3.]
 [15.]]
[[ 3.]
 [15.]]
[[6]
 [6]]
 
'''
