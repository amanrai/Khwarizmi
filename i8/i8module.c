#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "i8_ops.h"
#include <stdio.h>

typedef struct {
    PyObject_HEAD
    I8Array* array;
} PyI8Object;

static PyTypeObject PyI8Type;  // Forward declaration

static void
PyI8_dealloc(PyI8Object *self)
{
    printf("Deallocating PyI8Object\n");
    if (self->array) {
        i8_destroy(self->array);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
PyI8_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    printf("PyI8_new: Creating new PyI8Object\n");
    PyI8Object *self;
    PyObject *data_arg = NULL;
    Py_ssize_t size = 0;

    if (!PyArg_ParseTuple(args, "|O", &data_arg)) {
        printf("PyI8_new: Failed to parse arguments\n");
        return NULL;
    }
    
    self = (PyI8Object *) type->tp_alloc(type, 0);
    if (self == NULL) {
        printf("PyI8_new: Failed to allocate new object\n");
        return NULL;
    }

    self->array = NULL;  // Initialize to NULL

    if (data_arg == NULL) {
        printf("PyI8_new: No data argument provided\n");
        return (PyObject *) self;  // Return empty object
    }

    if (!PyList_Check(data_arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        Py_DECREF(self);
        return NULL;
    }

    size = PyList_Size(data_arg);
    printf("PyI8_new: Creating I8Array of size %zd\n", size);
    self->array = i8_create(size);
    if (self->array == NULL) {
        printf("PyI8_new: Failed to create I8Array\n");
        Py_DECREF(self);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PyList_GetItem(data_arg, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List items must be integers");
            Py_DECREF(self);
            return NULL;
        }
        long value = PyLong_AsLong(item);
        if (value < -128 || value > 127) {
            PyErr_SetString(PyExc_ValueError, "Integer out of range for int8");
            Py_DECREF(self);
            return NULL;
        }
        self->array->data[i] = (int8_t)value;
    }

    printf("PyI8_new: Successfully created PyI8Object\n");
    return (PyObject *) self;
}

static PyObject *
PyI8_add(PyI8Object *self, PyObject *other)
{
    printf("PyI8_add: Adding PyI8Objects\n");
    if (!PyObject_IsInstance(other, (PyObject *)&PyI8Type)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be an I8 object");
        return NULL;
    }
    
    PyI8Object *other_i8 = (PyI8Object *)other;
    
    if (self->array == NULL || other_i8->array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid I8 object");
        return NULL;
    }

    I8Array* result = i8_add(self->array, other_i8->array);
    if (result == NULL) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
        return NULL;
    }

    printf("PyI8_add: Creating new PyI8Object for result\n");
    PyObject *args = PyTuple_New(0);  // Create an empty tuple
    if (args == NULL) {
        i8_destroy(result);
        return NULL;
    }
    PyI8Object *py_result = (PyI8Object *)PyI8Type.tp_new(&PyI8Type, args, NULL);
    Py_DECREF(args);
    if (py_result == NULL) {
        printf("PyI8_add: Failed to create new PyI8Object\n");
        i8_destroy(result);
        return NULL;
    }

    py_result->array = result;
    printf("PyI8_add: Successfully created result PyI8Object\n");

    return (PyObject *) py_result;
}

static PyMethodDef PyI8_methods[] = {
    {"add", (PyCFunction) PyI8_add, METH_O, "Add two I8 arrays"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyI8Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "i8.I8",
    .tp_doc = "I8 objects",
    .tp_basicsize = sizeof(PyI8Object),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyI8_new,
    .tp_dealloc = (destructor) PyI8_dealloc,
    .tp_methods = PyI8_methods,
};

static PyModuleDef i8module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "i8",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_i8(void)
{
    printf("PyInit_i8: Initializing i8 module\n");
    PyObject *m;
    if (PyType_Ready(&PyI8Type) < 0) {
        printf("PyInit_i8: Failed to ready PyI8Type\n");
        return NULL;
    }

    m = PyModule_Create(&i8module);
    if (m == NULL) {
        printf("PyInit_i8: Failed to create module\n");
        return NULL;
    }

    Py_INCREF(&PyI8Type);
    if (PyModule_AddObject(m, "I8", (PyObject *) &PyI8Type) < 0) {
        printf("PyInit_i8: Failed to add I8 type to module\n");
        Py_DECREF(&PyI8Type);
        Py_DECREF(m);
        return NULL;
    }

    printf("PyInit_i8: Successfully initialized i8 module\n");
    return m;
}