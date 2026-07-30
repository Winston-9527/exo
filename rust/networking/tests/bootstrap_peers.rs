use std::net::{TcpListener, UdpSocket};
use std::time::Duration;

use networking::swarm::create_swarm;
use tokio::sync::mpsc;

#[tokio::test]
async fn create_swarm_rejects_invalid_bootstrap_endpoint() {
    let (_sender, receiver) = mpsc::channel(1);

    let result = create_swarm(
        "1",
        "static-bootstrap-test",
        receiver,
        52414,
        52413,
        vec!["not-a-zenoh-endpoint".to_owned()],
    )
    .await;

    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_swarms_connect_without_multicast_discovery() {
    let (first_listen_port, second_listen_port) = unused_tcp_ports();
    let (first_discovery_port, second_discovery_port) = unused_udp_ports();

    let (_first_sender, first_receiver) = mpsc::channel(1);
    let first_swarm = create_swarm(
        "1",
        "static-bootstrap-test",
        first_receiver,
        first_listen_port,
        first_discovery_port,
        vec![],
    )
    .await
    .expect("the listening swarm should start");

    let (_second_sender, second_receiver) = mpsc::channel(1);
    let second_swarm = create_swarm(
        "2",
        "static-bootstrap-test",
        second_receiver,
        second_listen_port,
        second_discovery_port,
        vec![format!("tcp/[::1]:{first_listen_port}")],
    )
    .await
    .expect("the dialing swarm should start");

    let subscriber = first_swarm
        .session
        .z
        .declare_subscriber("tests/static-bootstrap")
        .await
        .expect("the listening swarm should subscribe");
    let publisher = second_swarm
        .session
        .z
        .declare_publisher("tests/static-bootstrap")
        .await
        .expect("the dialing swarm should publish");
    let receive_payload = async {
        loop {
            tokio::select! {
                sample = subscriber.recv_async() => {
                    let sample = sample.expect("the subscriber should remain open");
                    break sample.payload().to_bytes().to_vec();
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    publisher
                        .put("connected")
                        .await
                        .expect("publishing the test payload should succeed");
                }
            }
        }
    };

    let payload = tokio::time::timeout(Duration::from_secs(5), receive_payload)
        .await
        .expect("static peers should exchange data without multicast");
    assert_eq!(payload.as_slice(), b"connected");
}

fn unused_tcp_ports() -> (u16, u16) {
    let first =
        TcpListener::bind("[::1]:0").expect("a first IPv6 loopback TCP port should be available");
    let second =
        TcpListener::bind("[::1]:0").expect("a second IPv6 loopback TCP port should be available");
    (
        first
            .local_addr()
            .expect("the first TCP listener should have a local address")
            .port(),
        second
            .local_addr()
            .expect("the second TCP listener should have a local address")
            .port(),
    )
}

fn unused_udp_ports() -> (u16, u16) {
    let first =
        UdpSocket::bind("[::1]:0").expect("a first IPv6 loopback UDP port should be available");
    let second =
        UdpSocket::bind("[::1]:0").expect("a second IPv6 loopback UDP port should be available");
    (
        first
            .local_addr()
            .expect("the first UDP socket should have a local address")
            .port(),
        second
            .local_addr()
            .expect("the second UDP socket should have a local address")
            .port(),
    )
}
