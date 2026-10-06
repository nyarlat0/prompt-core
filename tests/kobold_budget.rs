use prompt_core::{GenerationRequest, KoboldClient, Preset};
use serde_json::{Value, json};
use std::io::{Read, Write};

// No real model: verify the protocol and that overflow never reaches generate.
fn server(steps: Vec<(&'static str, Value, Value)>) -> (String, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let handle = std::thread::spawn(move || {
        for (path, expected, reply) in steps {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "missing request {path}"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(e) => panic!("{e}"),
                }
            };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            while !bytes.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                bytes.push(byte[0]);
            }
            let headers = String::from_utf8(bytes).unwrap();
            assert_eq!(
                headers.lines().next().unwrap().split_whitespace().nth(1),
                Some(path)
            );
            let length = headers
                .lines()
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .map(|(_, v)| v.trim().parse::<usize>().unwrap())
                .unwrap_or(0);
            let mut body = vec![0; length];
            stream.read_exact(&mut body).unwrap();
            let body: Value = if body.is_empty() {
                json!({})
            } else {
                serde_json::from_slice(&body).unwrap()
            };
            for (key, value) in expected.as_object().unwrap() {
                assert_eq!(&body[key], value, "{path}: {key}");
            }
            let reply = reply.to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).unwrap();
        }
    });
    (url, handle)
}

#[tokio::test]
async fn preset_and_native_calls_count_before_generating_and_share_reserve() {
    let (url, handle) = server(vec![
        (
            "/api/extra/true_max_context_length",
            json!({}),
            json!({"value":1024}),
        ),
        (
            "/api/extra/tokencount",
            json!({"prompt":"hello", "special":true}),
            json!({"value":10}),
        ),
        (
            "/api/v1/generate",
            json!({"prompt":"hello", "max_length":100, "max_context_length":1024}),
            json!({"results":[{"text":"one"}]}),
        ),
        (
            "/api/extra/tokencount",
            json!({"prompt":"world", "special":true}),
            json!({"value":10}),
        ),
        (
            "/api/v1/generate",
            json!({"prompt":"world", "max_length":256, "max_context_length":1024}),
            json!({"results":[{"text":"two"}]}),
        ),
    ]);
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        client
            .generate(
                "hello",
                &Preset::from_value(json!({"genamt":100})).unwrap(),
                &[]
            )
            .await
            .unwrap(),
        "one"
    );
    assert_eq!(
        client
            .generate_request(&GenerationRequest::new("world"))
            .await
            .unwrap()
            .text,
        "two"
    );
    handle.join().unwrap();
}

#[tokio::test]
async fn overflow_or_invalid_token_count_never_generates() {
    for count in [json!(1000), json!(-1), json!(0)] {
        let (url, handle) = server(vec![
            (
                "/api/extra/true_max_context_length",
                json!({}),
                json!({"value":1024}),
            ),
            (
                "/api/extra/tokencount",
                json!({"prompt":"hello"}),
                json!({"value":count}),
            ),
        ]);
        let client = KoboldClient::connect_with_client(
            url,
            reqwest::Client::builder().no_proxy().build().unwrap(),
        )
        .await
        .unwrap();
        let error = client
            .generate_request(&GenerationRequest::new("hello"))
            .await
            .unwrap_err();
        assert!(
            !error.to_string().contains("подключиться"),
            "unexpected generation attempt: {error}"
        );
        handle.join().unwrap();
    }
}
