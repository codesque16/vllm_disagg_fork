# Disaggregated Serving

This example contains scripts that demonstrate the disaggregated serving features of vLLM.

## Files

- `disagg_proxy_demo.py` - Demonstrates XpYd (X prefill instances, Y decode instances).
- `dflash_pd_example.sh` - Prefill/decode with DFlash Design A (`disagg_spec_role` producer/consumer + NixlConnector).
- `kv_events.sh` - Demonstrates KV cache event publishing.
- `mooncake_connector` - A proxy demo for MooncakeConnector.
