
# ====================================================================
#  You can add more layer shapes by creating new cnn_layers variables
# ====================================================================
# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride



# ---------- BELOW ARE THE PROVIDED SHAPES (CONV LAYERS of 3 DNN Models) --------------------
# Alex Net w/o grouping specified in http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf
# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride
# cnn_layers = [
#     (227, 227, 3, 1, 96, 11, 11, 1, 1, 4, 4),
#     (27, 27, 96, 1, 256, 5, 5, 2, 2, 1, 1),
#     (13, 13, 256, 1, 384, 3, 3, 1, 1, 1, 1),
#     (13, 13, 384, 1, 384, 3, 3, 1, 1, 1, 1),
#     (13, 13, 384, 1, 256, 3, 3, 1, 1, 1, 1),
#     ]

# VGG-1 Net Specified in [Shafiee, ISCA 2016]
# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride
cnn_layers = [
    (224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1),
    (112, 112, 64, 1, 128, 3, 3, 1, 1, 1, 1),
    (56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1),
    (56, 56, 256, 1, 256, 3, 3, 1, 1, 1, 1),
    (28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1),
    (28, 28, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    (14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    (14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    ]

# VGG-2 Net Specified in [Shafiee, ISCA 2016]
# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride
# cnn_layers = [
#     (224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1),
#     (224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1),
#     (112, 112, 64, 1, 128, 3, 3, 1, 1, 1, 1),
#     (112, 112, 128, 1, 128, 3, 3, 1, 1, 1, 1),
#     (56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1),
#     (56, 56, 256, 1, 256, 3, 3, 1, 1, 1, 1),
#     (56, 56, 256, 1, 256, 1, 1, 0, 0, 1, 1),
#     (28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1),
#     (28, 28, 512, 1, 512, 3, 3, 1, 1, 1, 1),
#     (28, 28, 512, 1, 512, 1, 1, 0, 0, 1, 1),
#     (14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1),
#     (14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1),
#     (14, 14, 512, 1, 256, 1, 1, 1, 1, 1, 1),
#     ]

# W, H, C, N, K, S, R, Wpad, Hpad, Wstride, Hstride
cnn_layers = [
    (224, 64, 3, 1, 64, 3, 3, 1, 1, 1, 1),
    (32, 32, 32, 1, 32, 3, 3, 1, 1, 1, 1),
    (56, 56, 16, 1, 16, 3, 3, 1, 1, 1, 1),
    (28, 28, 32, 1, 64, 3, 3, 1, 1, 1, 1),
    (14, 14, 128, 1, 4, 3, 3, 1, 1, 1, 1),
    (7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    (28, 28, 192, 1, 32, 5, 5, 2, 2, 1, 1),
    (28, 28, 192, 1, 64, 1, 1, 0, 0, 1, 1),
    (14, 14, 512, 1, 48, 5, 5, 2, 2, 1, 1),
    (14, 14, 512, 1, 192, 1, 1, 0, 0, 1, 1),
    (7, 7, 832, 1, 256, 1, 1, 0, 0, 1, 1),
    (7, 7, 832, 1, 128, 5, 5, 2, 2, 1, 1),
    (60, 6, 64, 1, 32, 3, 3, 1, 1, 1, 1),
    (151, 40, 1, 1, 32, 20, 5, 8, 8, 1, 1),
    (112, 112, 64, 1, 64, 1, 1, 0, 0, 1, 1),
    (56, 56, 2, 1, 256, 1, 1, 0, 0, 1, 1),
    (56, 56, 8, 1, 64, 1, 1, 0, 0, 1, 1),
    (56, 56, 4, 1, 128, 1, 1, 0, 0, 1, 1),
    (28, 28, 1, 1, 512, 1, 1, 0, 0, 1, 1),
    (28, 28, 4, 1, 128, 1, 1, 0, 0, 1, 1),
    (28, 28, 2, 1, 256, 1, 1, 0, 0, 2, 2),
    (14, 14, 2, 1, 256, 1, 1, 0, 0, 1, 1),
    (28, 28, 8, 1, 64, 1, 1, 0, 0, 1, 1),
    (14, 14, 1024, 1, 256, 1, 1, 0, 0, 1, 1),
    (14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
    (14, 14, 1024, 1, 512, 1, 1, 0, 0, 1, 1),
    (7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    (7, 7, 512, 1, 2048, 1, 1, 0, 0, 1, 1),
    (14, 14, 1024, 1, 2048, 1, 1, 0, 0, 1, 1),
    (7, 7, 2048, 1, 512, 1, 1, 0, 0, 1, 1),
    (16, 8, 4, 1, 4, 3, 3, 1, 1, 1, 1),
    (480, 48, 8, 1, 16, 3, 3, 1, 1, 1, 1),
    (32, 32, 1, 1, 6, 5, 5, 0, 0, 1, 1),
    (14, 14, 6, 1, 16, 5, 5, 0, 0, 1, 1),
    (1, 1, 400, 1, 120, 1, 1, 0, 0, 1, 1),
    (1, 1, 120, 1, 84, 1, 1, 0, 0, 1, 1),
    (227, 227, 3, 1, 96, 11, 11, 0, 0, 4, 4),
    (27, 27, 96, 1, 256, 5, 5, 2, 2, 1, 1),
    (13, 13, 256, 1, 384, 3, 3, 1, 1, 1, 1),
    (13, 13, 384, 1, 384, 3, 3, 1, 1, 1, 1),
    (13, 13, 384, 1, 256, 3, 3, 1, 1, 1, 1),
    (1, 1, 9216, 1, 4096, 1, 1, 0, 0, 1, 1),
    (1, 1, 4096, 1, 4096, 1, 1, 0, 0, 1, 1),
    (1, 1, 4096, 1, 1000, 1, 1, 0, 0, 1, 1),
    (32, 32, 32, 1, 32, 5, 5, 0, 0, 1, 1),
    (14, 14, 32, 1, 32, 5, 5, 0, 0, 1, 1)
]
