import sys
import json
from client import GeneralizedAdvantageEstimator

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-gae-generalized-advantage-estimation-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "calculate_gae_advantages",
                        "description": "Calculate GAE-lambda exponentially weighted advantages and target returns",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "rewards": {"type": "array", "items": {"type": "number"}},
                                "values": {"type": "array", "items": {"type": "number"}},
                                "next_value": {"type": "number", "default": 0.0},
                                "gamma": {"type": "number", "default": 0.99},
                                "lam": {"type": "number", "default": 0.95}
                            },
                            "required": ["rewards", "values"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "calculate_gae_advantages":
            r = args.get("rewards", [])
            v = args.get("values", [])
            nv = args.get("next_value", 0.0)
            gamma = args.get("gamma", 0.99)
            lam = args.get("lam", 0.95)
            res = GeneralizedAdvantageEstimator.compute_gae(r, v, nv, gamma=gamma, lam=lam)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res)}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
