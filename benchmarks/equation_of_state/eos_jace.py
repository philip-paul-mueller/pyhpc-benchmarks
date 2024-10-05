import jace
import jax
import jax.numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

v01 = 9.998420897506056e2
v02 = 2.839940833161907e0
v03 = -3.147759265588511e-2
v04 = 1.181805545074306e-3
v05 = -6.698001071123802e0
v06 = -2.986498947203215e-2
v07 = 2.327859407479162e-4
v08 = -3.988822378968490e-2
v09 = 5.095422573880500e-4
v10 = -1.426984671633621e-5
v11 = 1.645039373682922e-7
v12 = -2.233269627352527e-2
v13 = -3.436090079851880e-4
v14 = 3.726050720345733e-6
v15 = -1.806789763745328e-4
v16 = 6.876837219536232e-7
v17 = -3.087032500374211e-7
v18 = -1.988366587925593e-8
v19 = -1.061519070296458e-11
v20 = 1.550932729220080e-10
v21 = 1.0e0
v22 = 2.775927747785646e-3
v23 = -2.349607444135925e-5
v24 = 1.119513357486743e-6
v25 = 6.743689325042773e-10
v26 = -7.521448093615448e-3
v27 = -2.764306979894411e-5
v28 = 1.262937315098546e-7
v29 = 9.527875081696435e-10
v30 = -1.811147201949891e-11
v31 = -3.303308871386421e-5
v32 = 3.801564588876298e-7
v33 = -7.672876869259043e-9
v34 = -4.634182341116144e-11
v35 = 2.681097235569143e-12
v36 = 5.419326551148740e-6
v37 = -2.742185394906099e-5
v38 = -3.212746477974189e-7
v39 = 3.191413910561627e-9
v40 = -1.931012931541776e-12
v41 = -1.105097577149576e-7
v42 = 6.211426728363857e-10
v43 = -1.119011592875110e-10
v44 = -1.941660213148725e-11
v45 = -1.864826425365600e-14
v46 = 1.119522344879478e-14
v47 = -1.200507748551599e-15
v48 = 6.057902487546866e-17
rho0 = 1024.0


def gsw_dHdT(sa, ct, p):
    """
    d/dT of dynamic enthalpy, analytical derivative

    sa     : Absolute Salinity                               [g/kg]
    ct     : Conservative Temperature                        [deg C]
    p      : sea pressure                                    [dbar]
    """
    t1 = v45 * ct
    t2 = 0.2e1 * t1
    t3 = v46 * sa
    t4 = 0.5 * v12
    t5 = v14 * ct
    t7 = ct * (v13 + t5)
    t8 = 0.5 * t7
    t11 = sa * (v15 + v16 * ct)
    t12 = 0.5 * t11
    t13 = t4 + t8 + t12
    t15 = v19 * ct
    t19 = v17 + ct * (v18 + t15) + v20 * sa
    t20 = 1.0 / t19
    t24 = v47 + v48 * ct
    t25 = 0.5 * v13
    t26 = 1.0 * t5
    t27 = sa * v16
    t28 = 0.5 * t27
    t29 = t25 + t26 + t28
    t33 = t24 * t13
    t34 = t19 ** 2
    t35 = 1.0 / t34
    t37 = v18 + 2.0 * t15
    t38 = t35 * t37
    t48 = ct * (v44 + t1 + t3)
    t57 = v40 * ct
    t59 = ct * (v39 + t57)
    t64 = t13 ** 2
    t68 = t20 * t29
    t71 = t24 * t64
    t74 = v04 * ct
    t76 = ct * (v03 + t74)
    t79 = v07 * ct
    t82 = np.sqrt(sa)
    t83 = v11 * ct
    t85 = ct * (v10 + t83)
    t92 = (
        v01
        + ct * (v02 + t76)
        + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
    )
    t93 = v48 * t92
    t105 = (
        v02
        + t76
        + ct * (v03 + 2.0 * t74)
        + sa * (v06 + 2.0 * t79 + t82 * (v09 + t85 + ct * (v10 + 2.0 * t83)))
    )
    t106 = t24 * t105
    t107 = v44 + t2 + t3
    t110 = v43 + t48
    t117 = t24 * t92
    t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
    t123 = (
        v38
        + t59
        + ct * (v39 + 2.0 * t57)
        + sa * v42
        + (
            4.0 * v48 * t64 * t20
            + 8.0 * t33 * t68
            - 4.0 * t71 * t38
            - t93
            - t106
            - 2.0 * t107 * t13
            - 2.0 * t110 * t29
        )
        * t20
        - t120 * t35 * t37
    )
    t128 = t19 * p
    t130 = p * (1.0 * v12 + 1.0 * t7 + 1.0 * t11 + t128)
    t131 = 1.0 / t92
    t133 = 1.0 + t130 * t131
    t134 = np.log(t133)
    t143 = v37 + ct * (v38 + t59) + sa * (v41 + v42 * ct) + t120 * t20
    t152 = t37 * p
    t156 = t92 ** 2
    t165 = v25 * ct
    t167 = ct * (v24 + t165)
    t169 = ct * (v23 + t167)
    t175 = v30 * ct
    t177 = ct * (v29 + t175)
    t179 = ct * (v28 + t177)
    t185 = v35 * ct
    t187 = ct * (v34 + t185)
    t189 = ct * (v33 + t187)
    t199 = t13 * t20
    t217 = 2.0 * t117 * t199 - t110 * t92
    t234 = (
        v21
        + ct * (v22 + t169)
        + sa * (v26 + ct * (v27 + t179) + v36 * sa + t82 * (v31 + ct * (v32 + t189)))
        + t217 * t20
    )
    t241 = t64 - t92 * t19
    t242 = np.sqrt(t241)
    t243 = 1.0 / t242
    t244 = t4 + t8 + t12 - t242
    t245 = 1.0 / t244
    t247 = t4 + t8 + t12 + t242 + t128
    t248 = 1.0 / t247
    t249 = t242 * t245 * t248
    t252 = 1.0 + 2.0 * t128 * t249
    t253 = np.log(t252)
    t254 = t243 * t253
    t259 = t234 * t19 - t143 * t13
    t264 = t259 * t20
    t272 = 2.0 * t13 * t29 - t105 * t19 - t92 * t37
    t282 = t128 * t242
    t283 = t244 ** 2
    t287 = t243 * t272 / 2.0
    t292 = t247 ** 2
    t305 = (
        0.1e5
        * p
        * (
            v44
            + t2
            + t3
            - 2.0 * v48 * t13 * t20
            - 2.0 * t24 * t29 * t20
            + 2.0 * t33 * t38
            + 0.5 * v48 * p
        )
        * t20
        - 0.1e5 * p * (v43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38
        + 0.5e4 * t123 * t20 * t134
        - 0.5e4 * t143 * t35 * t134 * t37
        + 0.5e4
        * t143
        * t20
        * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        / t133
        + 0.5e4
        * (
            (
                v22
                + t169
                + ct * (v23 + t167 + ct * (v24 + 2.0 * t165))
                + sa
                * (
                    v27
                    + t179
                    + ct * (v28 + t177 + ct * (v29 + 2.0 * t175))
                    + t82 * (v32 + t189 + ct * (v33 + t187 + ct * (v34 + 2.0 * t185)))
                )
                + (
                    2.0 * t93 * t199
                    + 2.0 * t106 * t199
                    + 2.0 * t117 * t68
                    - 2.0 * t117 * t13 * t35 * t37
                    - t107 * t92
                    - t110 * t105
                )
                * t20
                - t217 * t35 * t37
            )
            * t19
            + t234 * t37
            - t123 * t13
            - t143 * t29
        )
        * t20
        * t254
        - 0.5e4 * t259 * t35 * t254 * t37
        - 0.25e4 * t264 / t242 / t241 * t253 * t272
        + 0.5e4
        * t264
        * t243
        * (
            2.0 * t152 * t249
            + t128 * t243 * t245 * t248 * t272
            - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
            - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)
        )
        / t252
    )

    return t305

@jace.jit(backend="gpu")
def gsw_dHdT_gpu(sa, ct, p):
    return gsw_dHdT(sa, ct, p)


@jace.jit(backend="cpu")
def gsw_dHdT_cpu(sa, ct, p):
    return gsw_dHdT(sa, ct, p)


def run(sa, ct, p, device="cpu"):
    wrapped = gsw_dHdT_cpu if device == "cpu" else gsw_dHdT_gpu
    lowered = wrapped.lower(sa, ct, p)
    compiled = lowered.compile()
    csdfg = compiled._compiled_sdfg.compiled_sdfg
    sdfg = csdfg._sdfg
    output_names = compiled._compiled_sdfg.output_names
    input_names = compiled._compiled_sdfg.input_names

    if not hasattr(csdfg, "_benchmark_args"):
        # The main issue we have is related to memory allocation, so we avoid that.
        #  This is a bit unfair to JAX, but since it does more in C++ it is not that
        #  much of a concern.
        # Thus we will allocate the memory once and then use `CompiledSDFG.fast_call()`
        #  and `CompiledSDFG._lastargs` to perform the call. However, we also have to
        #  ensure that the memory stays alive, th
        #  The simplest thing would be to attach the return value and arguments to the
        #  SDFG object (or any related object). However, this leads to memory corruption.
        #  This is caused because JaCe returns JAX array, if it returns NumPy arrays
        #  it does work. For that reason we have to reimplement the calling code!
        # NOTE: Also look at the setup function!
        import dace
        from dace import data as dace_data
        if True:
            sdfg_call_args: dict[str, Any] = {
                    input_name: arr  # This inherently assumes that we do not have a JAX array.
                    for input_name, arr in zip(input_names, [sa, ct, p])
            }
            sdfg_call_args.update({
                    output_name: dace_data.make_array_from_descriptor(sdfg.arrays[output_name])
                    for output_name in output_names
            })
            with dace.config.temporary_config():
                dace.Config.set("compiler", "allow_view_arguments", value=True)
                csdfg(**sdfg_call_args)
        else:
            # The simple thing that does not work
            _ret = compiled(sa, ct, p)
            sdfg_call_args = {
                    "sa": sa,
                    "ct": ct,
                    "p": p,
                    output_names[0]: _ret
            }
        setattr(csdfg, "_benchmark_args", sdfg_call_args)

    csdfg.fast_call(*csdfg._lastargs)
    return [csdfg._benchmark_args[output_name] for output_name in output_names]


def prepare_inputs(sa, ct, p, device):
    if device == "cpu":
        inputs = (sa, ct, p)
    elif device == "gpu":
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        inputs = [cp.asarray(k) for k in (sa, ct, p)]
        cp.cuda.stream.get_current_stream().synchronize()
    return inputs
