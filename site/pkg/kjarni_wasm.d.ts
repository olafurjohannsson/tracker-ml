/* tslint:disable */
/* eslint-disable */

export class WasmEncoder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    encode_file(text: string, source_path: string): any;
    static new(model_data: Uint8Array): WasmEncoder;
}

export class WasmFeedForward {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Classify a single input. Takes a pre-scaled feature vector (Float32Array).
     * Returns JSON: { label: 0|1, confidence: 0.0-1.0, scores: [f32, f32] }
     */
    classify(features: Float32Array): any;
    /**
     * Batch classify. Takes flat f32 array (n * input_dim) and returns JSON array of results.
     */
    classify_batch(features: Float32Array, count: number): any;
    /**
     * Load from raw safetensors bytes + config JSON string.
     * Expected safetensors keys:
     *   layer1.weight (hidden1, input_dim)
     *   layer1.bias   (hidden1,)
     *   layer2.weight (hidden2, hidden1)
     *   layer2.bias   (hidden2,)
     *   layer3.weight (output_dim, hidden2)
     *   layer3.bias   (output_dim,)
     */
    static load(safetensors_bytes: Uint8Array, config_json: string): WasmFeedForward;
}

export class WasmIndexBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    add_chunk(text: string, embedding: Float32Array, source: string, chunk_index: number): void;
    add_file(text: string, source_path: string): number;
    doc_count(): number;
    finish(): Uint8Array;
    static new(model_data: Uint8Array): WasmIndexBuilder;
}

export class WasmModel {
    free(): void;
    [Symbol.dispose](): void;
    encode(texts: string[], normalize: boolean): Float32Array;
    static from_quantized(data: Uint8Array): WasmModel;
    static from_type(model_type: WasmModelType): Promise<WasmModel>;
    constructor(weights_data: Uint8Array, config_json: string, tokenizer_json: string);
}

export enum WasmModelType {
    MiniLML6V2 = 0,
}

export class WasmReranker {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static load(data: Uint8Array): WasmReranker;
    rerank(query: string, documents: string[], limit: number): any;
    score(query: string, document: string): number;
}

export class WasmSearch {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    add_chunk(text: string, embedding: Float32Array, source: string, chunk_index: number): void;
    doc_count(): number;
    static load(model_data: Uint8Array, index_data: Uint8Array): WasmSearch;
    remove_file(source_path: string): number;
    save_index(): Uint8Array;
    search(query: string, limit: number): any;
    search_keywords(query: string, limit: number): any;
    search_semantic(query: string, limit: number): any;
    update_file(text: string, source_path: string): number;
}

export function init(): void;

export function set_debug_logging(enabled: boolean): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmencoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmfeedforward_free: (a: number, b: number) => void;
    readonly __wbg_wasmindexbuilder_free: (a: number, b: number) => void;
    readonly __wbg_wasmmodel_free: (a: number, b: number) => void;
    readonly __wbg_wasmreranker_free: (a: number, b: number) => void;
    readonly set_debug_logging: (a: number) => void;
    readonly wasmencoder_encode_file: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasmencoder_new: (a: number, b: number, c: number) => void;
    readonly wasmfeedforward_classify: (a: number, b: number, c: number, d: number) => void;
    readonly wasmfeedforward_classify_batch: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmfeedforward_load: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmindexbuilder_add_chunk: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly wasmindexbuilder_add_file: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasmindexbuilder_doc_count: (a: number) => number;
    readonly wasmindexbuilder_finish: (a: number, b: number) => void;
    readonly wasmindexbuilder_new: (a: number, b: number, c: number) => void;
    readonly wasmmodel_encode: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmmodel_from_quantized: (a: number, b: number, c: number) => void;
    readonly wasmmodel_from_type: (a: number) => number;
    readonly wasmmodel_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly wasmreranker_load: (a: number, b: number, c: number) => void;
    readonly wasmreranker_rerank: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly wasmreranker_score: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasmsearch_load: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmsearch_remove_file: (a: number, b: number, c: number) => number;
    readonly wasmsearch_save_index: (a: number, b: number) => void;
    readonly wasmsearch_search: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmsearch_search_keywords: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmsearch_search_semantic: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmsearch_update_file: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly init: () => void;
    readonly wasmsearch_doc_count: (a: number) => number;
    readonly __wbg_wasmsearch_free: (a: number, b: number) => void;
    readonly wasmsearch_add_chunk: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly __wasm_bindgen_func_elem_885: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_933: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_886: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
