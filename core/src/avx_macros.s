.macro badclass src, tmp
    vpsrlq \tmp, \src, 52
    vpandq \tmp, \tmp, qword ptr [{lane} + 24]{{1to8}}
    vpandq zmm14, \src, qword ptr [{lane} + 32]{{1to8}}
    vpcmpq k6, \tmp, qword ptr [{lane} + 40]{{1to8}}, 0
    vpcmpq k7, zmm14, qword ptr [{lane} + 40]{{1to8}}, 4
    kandw k6, k6, k7
    vpcmpq k7, \tmp, qword ptr [{lane} + 24]{{1to8}}, 0
    korw k6, k6, k7
.endm
.macro push8 src
    vcmppd k2, \src, qword ptr [{lane} + 40]{{1to8}}, 12
    kortestw k2, k2
    jz 8\@f
    vandnpd zmm7, zmm15, zmm0
    vandnpd zmm8, zmm15, \src
    vcmppd k3, zmm7, zmm8, 29
    vblendmpd zmm7{{k3}}, \src, zmm0
    vblendmpd zmm8{{k3}}, zmm0, \src
    vaddpd zmm9, zmm7, zmm8
    vsubpd zmm10, zmm7, zmm9
    vaddpd zmm10, zmm10, zmm8
    vaddpd zmm11, zmm1, zmm10
    badclass \src, zmm12
    kmovw k4, k6
    badclass zmm9, zmm12
    korw k4, k4, k6
    badclass zmm10, zmm12
    korw k4, k4, k6
    badclass zmm11, zmm12
    korw k4, k4, k6
    vandnpd zmm12, zmm15, zmm1
    vandnpd zmm13, zmm15, zmm10
    vaddpd zmm12, zmm12, zmm13
    vmulpd zmm13, zmm12, qword ptr [{lane} + 8]{{1to8}}
    vaddpd zmm14, zmm2, zmm13
    vmulpd zmm7, zmm14, qword ptr [{lane} + 16]{{1to8}}
    vaddpd zmm7, zmm14, zmm7
    vpsrlq zmm14, zmm7, 52
    vpandq zmm14, zmm14, qword ptr [{lane} + 24]{{1to8}}
    vpcmpq k5, zmm14, qword ptr [{lane} + 24]{{1to8}}, 0
    vcmppd k6, zmm12, qword ptr [{lane} + 40]{{1to8}}, 12
    vcmppd k7, zmm13, qword ptr [{lane} + 40]{{1to8}}, 0
    kandw k6, k6, k7
    korw k4, k4, k5
    korw k4, k4, k6
    kandw k4, k4, k2
    korw k1, k1, k4
    vblendmpd zmm0{{k2}}, zmm0, zmm9
    vblendmpd zmm1{{k2}}, zmm1, zmm11
    vblendmpd zmm2{{k2}}, zmm2, zmm7
8\@:
.endm
