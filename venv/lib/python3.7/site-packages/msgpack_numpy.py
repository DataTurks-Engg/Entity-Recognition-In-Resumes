#!/usr/bin/env python

"""
Support for serialization of numpy data types with msgpack.
"""

# Copyright (c) 2013-2018, Lev E. Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import os
import sys
import functools

import numpy as np
import msgpack

from msgpack import Packer as _Packer, Unpacker as _Unpacker, \
    unpack as _unpack, unpackb as _unpackb

def encode(obj, chain=None):
    """
    Data encoder for serializing numpy data types.
    """

    if isinstance(obj, np.ndarray):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if obj.dtype.kind == 'V':
            kind = b'V'
            descr = obj.dtype.descr
        else:
            kind = b''
            descr = obj.dtype.str
        return {b'nd': True,
                b'type': descr,
                b'kind': kind,
                b'shape': obj.shape,
                b'data': obj.tobytes()}
    elif isinstance(obj, (np.bool_, np.number)):
        return {b'nd': False,
                b'type': obj.dtype.str,
                b'data': obj.tobytes()}
    elif isinstance(obj, complex):
        return {b'complex': True,
                b'data': obj.__repr__()}
    else:
        return obj if chain is None else chain(obj)

def tostr(x):
    if sys.version_info >= (3, 0):
        if isinstance(x, bytes):
            return x.decode()
        else:
            return str(x)
    else:
        return x

def decode(obj, chain=None):
    """
    Decoder for deserializing numpy data types.
    """

    try:
        if b'nd' in obj:
            if obj[b'nd'] is True:

                # Check if b'kind' is in obj to enable decoding of data
                # serialized with older versions (#20):
                if b'kind' in obj and obj[b'kind'] == b'V':
                    descr = [tuple(tostr(t) if type(t) is bytes else t for t in d) \
                             for d in obj[b'type']]
                else:
                    descr = obj[b'type']
                return np.frombuffer(obj[b'data'],
                            dtype=np.dtype(descr)).reshape(obj[b'shape'])
            else:
                descr = obj[b'type']
                return np.frombuffer(obj[b'data'],
                            dtype=np.dtype(descr))[0]
        elif b'complex' in obj:
            return complex(tostr(obj[b'data']))
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)

# Maintain support for msgpack < 0.4.0:
if msgpack.version < (0, 4, 0):
    class Packer(_Packer):
        def __init__(self, default=None,
                     encoding='utf-8',
                     unicode_errors='strict',
                     use_single_float=False,
                     autoreset=1):
            default = functools.partial(encode, chain=default)
            super(Packer, self).__init__(default=default,
                                         encoding=encoding,
                                         unicode_errors=unicode_errors,
                                         use_single_float=use_single_float,
                                         autoreset=autoreset)
    class Unpacker(_Unpacker):
        def __init__(self, file_like=None, read_size=0, use_list=None,
                     object_hook=None,
                     object_pairs_hook=None, list_hook=None, encoding='utf-8',
                     unicode_errors='strict', max_buffer_size=0):
            object_hook = functools.partial(decode, chain=object_hook)
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           encoding=encoding,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size)

else:
    class Packer(_Packer):
        def __init__(self, default=None,
                     encoding='utf-8',
                     unicode_errors='strict',
                     use_single_float=False,
                     autoreset=1,
                     use_bin_type=0):
            default = functools.partial(encode, chain=default)
            super(Packer, self).__init__(default=default,
                                         encoding=encoding,
                                         unicode_errors=unicode_errors,
                                         use_single_float=use_single_float,
                                         autoreset=autoreset,
                                         use_bin_type=use_bin_type)

    class Unpacker(_Unpacker):
        def __init__(self, file_like=None, read_size=0, use_list=None,
                     object_hook=None,
                     object_pairs_hook=None, list_hook=None, encoding=None,
                     unicode_errors='strict', max_buffer_size=0,
                     ext_hook=msgpack.ExtType):
            object_hook = functools.partial(decode, chain=object_hook)
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           encoding=encoding,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size,
                                           ext_hook=ext_hook)

def pack(o, stream, **kwargs):
    """
    Pack an object and write it to a stream.
    """

    packer = Packer(**kwargs)
    stream.write(packer.pack(o))

def packb(o, **kwargs):
    """
    Pack an object and return the packed bytes.
    """

    return Packer(**kwargs).pack(o)

def unpack(stream, **kwargs):
    """
    Unpack a packed object from a stream.
    """

    object_hook = kwargs.get('object_hook')
    kwargs['object_hook'] = functools.partial(decode, chain=object_hook)
    return _unpack(stream, **kwargs)

def unpackb(packed, **kwargs):
    """
    Unpack a packed object.
    """

    object_hook = kwargs.get('object_hook')
    kwargs['object_hook'] = functools.partial(decode, chain=object_hook)
    return _unpackb(packed, **kwargs)

load = unpack
loads = unpackb
dump = pack
dumps = packb

def patch():
    """
    Monkey patch msgpack module to enable support for serializing numpy types.
    """

    setattr(msgpack, 'Packer', Packer)
    setattr(msgpack, 'Unpacker', Unpacker)
    setattr(msgpack, 'load', unpack)
    setattr(msgpack, 'loads', unpackb)
    setattr(msgpack, 'dump', pack)
    setattr(msgpack, 'dumps', packb)
    setattr(msgpack, 'pack', pack)
    setattr(msgpack, 'packb', packb)
    setattr(msgpack, 'unpack', unpack)
    setattr(msgpack, 'unpackb', unpackb)

if __name__ == '__main__':
    try:
        range = xrange # Python 2
    except NameError:
        pass # Python 3

    from unittest import main, TestCase, TestSuite
    from numpy.testing import assert_equal, assert_array_equal

    class ThirdParty(object):

        def __init__(self, foo=b'bar'):
            self.foo = foo

        def __eq__(self, other):
            return isinstance(other, ThirdParty) and self.foo == other.foo


    class test_numpy_msgpack(TestCase):
        def setUp(self):
             patch()
             
        def encode_decode(self, x, use_bin_type=False, encoding=None):
            x_enc = msgpack.packb(x, use_bin_type=use_bin_type)
            return msgpack.unpackb(x_enc, encoding=encoding)

        def encode_thirdparty(self, obj):
            return dict(__thirdparty__=True, foo=obj.foo)

        def decode_thirdparty(self, obj):
            if b'__thirdparty__' in obj:
                return ThirdParty(foo=obj[b'foo'])
            return obj

        def encode_decode_thirdparty(self, x, use_bin_type=False, encoding=None):
            x_enc = msgpack.packb(x, default=self.encode_thirdparty,
                                  use_bin_type=use_bin_type)
            return msgpack.unpackb(x_enc, object_hook=self.decode_thirdparty,
                                   encoding=encoding)

        def test_bin(self):
            # Since bytes == str in Python 2.7, the following
            # should pass on both 2.7 and 3.*
            assert_equal(type(self.encode_decode(b'foo')), bytes)
                
        def test_str(self):
            assert_equal(type(self.encode_decode('foo')), bytes)
            if sys.version_info.major == 2:
                assert_equal(type(self.encode_decode(u'foo')), str)

                # Test non-default string encoding/decoding:
                assert_equal(type(self.encode_decode(u'foo', True, 'utf=8')), unicode)
                
        def test_numpy_scalar_bool(self):
            x = np.bool_(True)
            x_rec = self.encode_decode(x)
            assert_equal(x, x_rec)
            assert_equal(type(x), type(x_rec))
            x = np.bool_(False)
            x_rec = self.encode_decode(x)
            assert_equal(x, x_rec)
            assert_equal(type(x), type(x_rec))
            
        def test_numpy_scalar_float(self):
            x = np.float32(np.random.rand())
            x_rec = self.encode_decode(x)
            assert_equal(x, x_rec)
            assert_equal(type(x), type(x_rec))
            
        def test_numpy_scalar_complex(self):
            x = np.complex64(np.random.rand()+1j*np.random.rand())
            x_rec = self.encode_decode(x)
            assert_equal(x, x_rec)
            assert_equal(type(x), type(x_rec))
            
        def test_scalar_float(self):
            x = np.random.rand()
            x_rec = self.encode_decode(x)
            assert_equal(x, x_rec)
            assert_equal(type(x), type(x_rec))
            
        def test_scalar_complex(self):
            x = np.random.rand()+1j*np.random.rand()
            x_rec = self.encode_decode(x)
            assert_equal(x, x_rec)
            assert_equal(type(x), type(x_rec))
            
        def test_list_numpy_float(self):
            x = [np.float32(np.random.rand()) for i in range(5)]
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_array_equal([type(e) for e in x],
                               [type(e) for e in x_rec])
            
        def test_list_numpy_float_complex(self):
            x = [np.float32(np.random.rand()) for i in range(5)] + \
              [np.complex128(np.random.rand()+1j*np.random.rand()) for i in range(5)]
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_array_equal([type(e) for e in x],
                               [type(e) for e in x_rec])
            
        def test_list_float(self):
            x = [np.random.rand() for i in range(5)]
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_array_equal([type(e) for e in x],
                               [type(e) for e in x_rec])
            
        def test_list_float_complex(self):
            x = [(np.random.rand()+1j*np.random.rand()) for i in range(5)]
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_array_equal([type(e) for e in x],
                               [type(e) for e in x_rec])
            
        def test_list_str(self):
            x = [b'x'*i for i in range(5)]
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_array_equal([type(e) for e in x_rec], [bytes]*5)
            
        def test_dict_float(self):
            x = {b'foo': 1.0, b'bar': 2.0}
            x_rec = self.encode_decode(x)
            assert_array_equal(sorted(x.values()), sorted(x_rec.values()))
            assert_array_equal([type(e) for e in sorted(x.values())],
                               [type(e) for e in sorted(x_rec.values())])
            assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
            assert_array_equal([type(e) for e in sorted(x.keys())],
                               [type(e) for e in sorted(x_rec.keys())])
            
        def test_dict_complex(self):
            x = {b'foo': 1.0+1.0j, b'bar': 2.0+2.0j}
            x_rec = self.encode_decode(x)
            assert_array_equal(sorted(x.values(), key=np.linalg.norm),
                               sorted(x_rec.values(), key=np.linalg.norm))
            assert_array_equal([type(e) for e in sorted(x.values(), key=np.linalg.norm)],
                               [type(e) for e in sorted(x_rec.values(), key=np.linalg.norm)])
            assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
            assert_array_equal([type(e) for e in sorted(x.keys())],
                               [type(e) for e in sorted(x_rec.keys())])

        def test_dict_str(self):
            x = {b'foo': b'xxx', b'bar': b'yyyy'}
            x_rec = self.encode_decode(x)
            assert_array_equal(sorted(x.values()), sorted(x_rec.values()))
            assert_array_equal([type(e) for e in sorted(x.values())],
                               [type(e) for e in sorted(x_rec.values())])
            assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
            assert_array_equal([type(e) for e in sorted(x.keys())],
                               [type(e) for e in sorted(x_rec.keys())])
            
        def test_dict_numpy_float(self):
            x = {b'foo': np.float32(1.0), b'bar': np.float32(2.0)}
            x_rec = self.encode_decode(x)
            assert_array_equal(sorted(x.values()), sorted(x_rec.values()))
            assert_array_equal([type(e) for e in sorted(x.values())],
                               [type(e) for e in sorted(x_rec.values())])
            assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
            assert_array_equal([type(e) for e in sorted(x.keys())],
                               [type(e) for e in sorted(x_rec.keys())])

        def test_dict_numpy_complex(self):
            x = {b'foo': np.complex128(1.0+1.0j), b'bar': np.complex128(2.0+2.0j)}
            x_rec = self.encode_decode(x)
            assert_array_equal(sorted(x.values(), key=np.linalg.norm),
                               sorted(x_rec.values(), key=np.linalg.norm))
            assert_array_equal([type(e) for e in sorted(x.values(), key=np.linalg.norm)],
                               [type(e) for e in sorted(x_rec.values(), key=np.linalg.norm)])
            assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
            assert_array_equal([type(e) for e in sorted(x.keys())],
                               [type(e) for e in sorted(x_rec.keys())])
            
        def test_numpy_array_float(self):
            x = np.random.rand(5).astype(np.float32)
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_equal(x.dtype, x_rec.dtype)
            
        def test_numpy_array_complex(self):
            x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_equal(x.dtype, x_rec.dtype)
            
        def test_numpy_array_float_2d(self):
            x = np.random.rand(5,5).astype(np.float32)
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_equal(x.dtype, x_rec.dtype)
            
        def test_numpy_array_str(self):
            x = np.array([b'aaa', b'bbbb', b'ccccc'])
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_equal(x.dtype, x_rec.dtype)
            
        def test_numpy_array_mixed(self):
            x = np.array([(1, 2, b'a', [1.0, 2.0])],
                         np.dtype([('arg0', np.uint32),
                                   ('arg1', np.uint32),
                                   ('arg2', 'S1'),
                                   ('arg3', np.float32, (2,))]))
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_equal(x.dtype, x_rec.dtype)
            
        def test_list_mixed(self):
            x = [1.0, np.float32(3.5), np.complex128(4.25), b'foo']
            x_rec = self.encode_decode(x)
            assert_array_equal(x, x_rec)
            assert_array_equal([type(e) for e in x],
                               [type(e) for e in x_rec])

        def test_chain(self):
            x = ThirdParty(foo=b'test marshal/unmarshal')
            x_rec = self.encode_decode_thirdparty(x)
            self.assertEqual(x, x_rec)

    main()

