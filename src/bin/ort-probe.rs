//! Probe and print ONNX Runtime execution providers.

use edge_ort_runtime::runtime::probe_providers;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_writer(std::io::stderr)
        .init();

    println!("ONNX Runtime execution providers (priority chain):\n");
    let providers = probe_providers();
    for (i, p) in providers.iter().enumerate() {
        let status = if p.available {
            "AVAILABLE"
        } else if p.platform_ok {
            "built-in support missing / not loaded"
        } else {
            "unsupported on this platform"
        };
        println!(
            "  {}. {:<12} {:<28} {}",
            i + 1,
            p.preference.as_str(),
            p.name,
            status
        );
    }
    println!();
}
