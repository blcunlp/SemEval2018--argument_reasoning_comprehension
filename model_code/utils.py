"""Numpy-related utilities."""
from __future__ import absolute_import

import sys
import numpy as np
from six.moves import range
from six.moves import zip
import os

PAD_ID = 0


def to_categorical(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def binary_logloss(p, y):
    epsilon = 1e-15
    p = np.maximum(epsilon, p)
    p = np.minimum(1 - epsilon, p)
    res = sum(y * np.log(p) + np.subtract(1, y) * np.log(np.subtract(1, p)))
    res *= -1.0 / len(y)
    return res


def multiclass_logloss(p, y):
    npreds = [p[i][y[i] - 1] for i in range(len(y))]
    score = -(1. / len(y)) * np.sum(np.log(npreds))
    return score


def accuracy(p, y):
    return np.mean([a == b for a, b in zip(p, y)])


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def convert_kernel(kernel, dim_ordering=None):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    # Arguments
        kernel: Numpy array (4D or 5D).
        dim_ordering: the data format.

    # Returns
        The converted kernel.

    # Raises
        ValueError: in case of invalid kernel shape or invalid dim_ordering.
    """
    if dim_ordering is None:
        dim_ordering = K.image_dim_ordering()
    if not 4 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)

    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    if dim_ordering == 'th':  # (out_depth, input_depth, ...)
        slices[:2] = no_flip
    elif dim_ordering == 'tf':  # (..., input_depth, out_depth)
        slices[-2:] = no_flip
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    return np.copy(kernel[slices])


def conv_output_length(input_length, filter_size,
                       border_mode, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        border_mode: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'full'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, border_mode, stride):
    """Determines input length of a convolution given output length.

    # Arguments
        output_length: integer.
        filter_size: integer.
        border_mode: one of "same", "valid", "full".
        stride: integer.

    # Returns
        The input length (integer).
    """
    if output_length is None:
        return None
    assert border_mode in {'same', 'valid', 'full'}
    if border_mode == 'same':
        pad = filter_size // 2
    elif border_mode == 'valid':
        pad = 0
    elif border_mode == 'full':
        pad = filter_size - 1
    return (output_length - 1) * stride - 2 * pad + filter_size


def get_batches(n, batch_size):
    # batch size [(start,end),(start,end),...,(start,end)] tuple list
    num_batches = int(np.ceil(n / float(batch_size)))
    return [(i * batch_size, min(n, (i + 1) * batch_size )) for i in range(0, num_batches)]


def data_summary_from_list2all(istest, warrant0_list, warrant1_list, reason_list, claim_list, debate_meta_data_list,correct_label_w0_or_w1_list,id_list):
    data_all = []
    if istest:
        for i in range(len(warrant0_list)):
            sentence = []
            sentence.append(warrant0_list[i])
            sentence.append(warrant1_list[i])
            sentence.append(reason_list[i])
            sentence.append(claim_list[i])
            sentence.append(debate_meta_data_list[i])
            sentence.append(correct_label_w0_or_w1_list)
            sentence.append(id_list[i])
            data_all.append(sentence)
        return data_all
    else:
        for i in range(len(warrant0_list)):
            sentence = []
            sentence.append(warrant0_list[i])
            sentence.append(warrant1_list[i])
            sentence.append(reason_list[i])
            sentence.append(claim_list[i])
            sentence.append(debate_meta_data_list[i])
            sentence.append(correct_label_w0_or_w1_list[i])
            sentence.append(id_list[i])
            data_all.append(sentence)
        return data_all




def prepare_batch_data(istest, data, start_id, end_id, max_len):
    # padding to given max sentences len
    batch_data = data[start_id: end_id]
    batch_size = len(batch_data)

    worrant0_list=[]
    worrant1_list=[]
    reason_list=[]
    claim_list=[]
    debate_list=[]
    label_list=[]
    id_list=[]
    if istest:
        label = []
        for idx, sentence in enumerate(batch_data):
            worrant0_list.append(sentence[0])
            worrant1_list.append(sentence[1])
            reason_list.append(sentence[2])
            claim_list.append(sentence[3])
            debate_list.append(sentence[4])
            label_list.append(label)
            id_list.append(sentence[6])
    else:
        for idx, sentence in enumerate(batch_data):
            worrant0_list.append(sentence[0])
            worrant1_list.append(sentence[1])
            reason_list.append(sentence[2])
            claim_list.append(sentence[3])
            debate_list.append(sentence[4])
            label_list.append(sentence[5])
            id_list.append(sentence[6])

    lengths_s1 = [len(s) for s in worrant0_list]
    lengths_s2 = [len(s) for s in worrant1_list]
    lengths_s3 = [len(s) for s in reason_list]
    lengths_s4 = [len(s) for s in claim_list]
    lengths_s5 = [len(s) for s in debate_list]


    w0 = np.ones((batch_size, max_len)).astype('int32') * PAD_ID
    w1 = np.ones((batch_size, max_len)).astype('int32') * PAD_ID
    r = np.ones((batch_size, max_len)).astype('int32') * PAD_ID
    c = np.ones((batch_size, max_len)).astype('int32') * PAD_ID
    d = np.ones((batch_size, max_len)).astype('int32') * PAD_ID

    for idx, [s_1, s_2, s_3, s_4, s_5] in enumerate(
                zip(worrant0_list, worrant1_list, reason_list, claim_list, debate_list)):
        w0[idx, :lengths_s1[idx]] = s_1
        w1[idx, :lengths_s2[idx]] = s_2
        r[idx, :lengths_s3[idx]] = s_3
        c[idx, :lengths_s4[idx]] = s_4
        d[idx, :lengths_s5[idx]] = s_5
    #print("label_list",label_list)
    #input()
    return w0, w1, r, c, d,label_list,id_list, np.array(lengths_s1), np.array(lengths_s2), np.array(lengths_s3), np.array(lengths_s4), np.array(lengths_s5)

def print_params(flags):
    params_dict = vars(flags)
    for k,v in params_dict.items():
        print('{}={}'.format(k,v))

def result_analysis(gold_labels_dev,predicted_labels_dev,ids_dev):
    #try:
    #    os.remove("./model_analysis_ids")
    #except:
    #    pass
    #try:
    #    os.remove(sys.argv[2])
    #except:
    #    pass

    #result_analysis_file = open("./model_analysis_ids",'w')
    #answer_file = open(sys.argv[2],'w')
    answer_file = open("dev_answer",'w')
    #good_ids = []
    #wrong_ids = []
    #print ("len:",len(gold_labels_dev))
    for i in range(len(gold_labels_dev)):
        #print("good:",gold_labels_dev[i])
        #print("pre:",predicted_labels_dev[i])
        answer_file.write(str(ids_dev[i])+'\t'+str(predicted_labels_dev[i])+'\n')
        #if gold_labels_dev[i] == predicted_labels_dev[i]:
        #    good_ids.append(''.join(ids_dev[i]))
        #else:
        #    wrong_ids.append(''.join(ids_dev[i]))
    #result_analysis_file.write("Good_ids\t" + str(good_ids) + "\n")
    #result_analysis_file.write("Wrong_ids\t" + str(wrong_ids) + "\n")
    #result_analysis_file.close()
    #print("good_num:",len(good_ids))
    #print("wrong_num:",len(wrong_ids))

def result_print(predicted_labels_dev,ids_dev):
    #try:
    #    os.remove("./answer.txt")
    #except:
    #    pass

    #answer_file = open(sys.argv[3],'w')
    answer_file = open("test_answer",'w')
    for i in range(len(predicted_labels_dev)):
        #print("pre:",predicted_labels_dev[i])
        answer_file.write(str(ids_dev[i])+'\t'+str(predicted_labels_dev[i])+'\n')


