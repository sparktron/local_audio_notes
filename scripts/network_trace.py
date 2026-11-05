"""Summarize an strace network trace written by observe_network.sh.

    python3 -I scripts/network_trace.py TRACE COMMAND_EXIT_STATUS

Prints every distinct endpoint as JSON. Exit status 0 when no IP traffic goes anywhere except
loopback port 11434 (Ollama) or a loopback port the traced command bound itself; 1 otherwise.
"""

import json
import re
import sys

CALLS = re.compile(r"^\d+\s+(connect|bind|sendto|sendmsg|sendmmsg)\(")
# Quoted strings are payload unless they are an address or socket path; blank the rest so
# data such as a forged '{sa_family=AF_INET, ...}' inside a buffer is never parsed.
QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"(?:\.\.\.)?')
ADDRESS_PREFIXES = ("inet_addr(", "AF_INET6, ", "sun_path=", "sun_path=@")
# sendto carries the destination as its fifth argument, sendmsg/sendmmsg under msg_name, and
# connect/bind as the second: every sockaddr in the call is a destination or local address.
SOCKADDR = re.compile(r"\{sa_family=(AF_\w+)([^{}]*)\}")
PORT = re.compile(r"sin6?_port=htons\((\d+)\)")
IPV4 = re.compile(r'sin_addr=inet_addr\("([^"]+)"\)')
IPV6 = re.compile(r'inet_pton\(AF_INET6, "([^"]+)"')
UNIX = re.compile(r'sun_path=@?"([^"]*)"')
LOOPBACK = ("127.", "::1")


def blank_payloads(line: str) -> str:
    def replace(match: re.Match) -> str:
        before = line[: match.start()]
        return match.group(0) if before.endswith(ADDRESS_PREFIXES) else '""'

    return QUOTED.sub(replace, line)


def summarize(lines) -> dict:
    seen: dict[str, int] = {}
    bound: set[int] = set()
    for line in lines:
        match = CALLS.match(line)
        if not match:
            continue
        call = match.group(1)
        for family, rest in SOCKADDR.findall(blank_payloads(line)):
            if family in {"AF_INET", "AF_INET6"}:
                port, address = (
                    PORT.search(rest),
                    (IPV4 if family == "AF_INET" else IPV6).search(rest),
                )
                if not port or not address:
                    key = f"{family} unparsed:0"  # never silently dropped
                elif call == "bind":
                    bound.add(int(port.group(1)))
                    continue
                else:
                    key = f"{family} {address.group(1)}:{port.group(1)}"
            elif family == "AF_UNIX":
                path = UNIX.search(rest)
                key = f"AF_UNIX {path.group(1) if path else '?'}"
            else:
                key = family  # e.g. AF_NETLINK interface enumeration (kernel, not network)
            seen[key] = seen.get(key, 0) + 1
    allowed = {11434} | bound
    violations = sorted(
        k
        for k in seen
        if k.startswith("AF_INET")
        and not (k.split()[1].startswith(LOOPBACK) and int(k.rsplit(":", 1)[1]) in allowed)
    )
    return {
        "endpoints": dict(sorted(seen.items())),
        "loopback_ports_bound_by_command": sorted(bound),
        "unexpected_ip_endpoints": violations,
    }


def main() -> None:
    trace, status = sys.argv[1], int(sys.argv[2])
    with open(trace, encoding="utf-8", errors="replace") as handle:
        result = {"command_exit_status": status, **summarize(handle)}
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["unexpected_ip_endpoints"] else 0)


if __name__ == "__main__":
    main()
