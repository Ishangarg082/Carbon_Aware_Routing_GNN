from ns import ns
helpers = ["UdpServerHelper","UdpClientHelper","OnOffHelper","UdpEchoServerHelper","BulkSendHelper","PacketSinkHelper"]
for h in helpers:
    print(h, hasattr(ns, h))
