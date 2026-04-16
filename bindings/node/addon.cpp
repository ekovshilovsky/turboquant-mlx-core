#include <node_api.h>
#include <cstring>
#include "turboquant_c/turboquant_c.h"

/*
 * TurboQuant-MLX Node.js N-API addon.
 *
 * Exposes a thin binding layer over the C API for use in Colima-hosted
 * containers and local tooling. The primary integration path for containers
 * communicating with SwiftLM is HTTP; these bindings cover the subset of
 * operations that benefit from direct in-process access.
 *
 * Exported surface:
 *   version()                 — returns the library version string
 *   validateModel(path)       — returns true if the model at path loads cleanly
 *   convertModel(options)     — stub; requires a C API addition (see TODO)
 */

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

/*
 * Throw a typed Error into the JS environment and return undefined.
 * Used to surface C-level failures as catchable JS exceptions.
 */
static napi_value ThrowTypeError(napi_env env, const char* message) {
    napi_throw_type_error(env, nullptr, message);
    return nullptr;
}

/* --------------------------------------------------------------------------
 * version() -> string
 *
 * Delegates to tq_version(), which returns a static string owned by the
 * library. No memory management is required on the caller side.
 * -------------------------------------------------------------------------- */
static napi_value GetVersion(napi_env env, napi_callback_info /*info*/) {
    napi_value result;
    napi_create_string_utf8(env, tq_version(), NAPI_AUTO_LENGTH, &result);
    return result;
}

/* --------------------------------------------------------------------------
 * validateModel(path: string) -> boolean
 *
 * Attempts to load the model at the given directory path via the C API.
 * Returns true if the load succeeds, false if it fails. The model handle
 * is freed immediately — this call is a probe, not a long-lived load.
 * -------------------------------------------------------------------------- */
static napi_value ValidateModel(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 1) {
        return ThrowTypeError(env, "validateModel requires a path argument");
    }

    /* Extract the path string from the JS argument. */
    size_t path_len = 0;
    napi_get_value_string_utf8(env, args[0], nullptr, 0, &path_len);

    char* path = new char[path_len + 1];
    napi_get_value_string_utf8(env, args[0], path, path_len + 1, &path_len);

    tq_model_t model = tq_model_load(path);
    delete[] path;

    bool valid = (model != nullptr);
    if (model != nullptr) {
        tq_model_free(model);
    }

    napi_value result;
    napi_get_boolean(env, valid, &result);
    return result;
}

/* --------------------------------------------------------------------------
 * convertModel(options: object) -> boolean
 *
 * TODO: The C API does not yet expose a convert_model entry point. This
 * function is a documented stub that signals the gap clearly rather than
 * silently returning incorrect results. Add tq_model_convert() to the C API
 * (turboquant_c.h / turboquant_c.cpp) and update this binding accordingly.
 * -------------------------------------------------------------------------- */
static napi_value ConvertModel(napi_env env, napi_callback_info /*info*/) {
    napi_throw_error(
        env,
        "NOT_IMPLEMENTED",
        "convertModel is not yet implemented: the C API does not expose "
        "tq_model_convert(). Add the entry point to turboquant_c.h and "
        "reimplement this binding.");
    return nullptr;
}

/* --------------------------------------------------------------------------
 * Module initialisation
 * -------------------------------------------------------------------------- */
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor descriptors[] = {
        {"version",       nullptr, GetVersion,    nullptr, nullptr, nullptr, napi_default, nullptr},
        {"validateModel", nullptr, ValidateModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"convertModel",  nullptr, ConvertModel,  nullptr, nullptr, nullptr, napi_default, nullptr},
    };

    napi_define_properties(
        env,
        exports,
        sizeof(descriptors) / sizeof(descriptors[0]),
        descriptors);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
