<?php
// /home/digitala/public_html/insights/upload_handler.php

// 1. --- SECURITY: Define the secret key ---
$EXPECTED_API_KEY = 'D8TU5PXNA895NU5ZXUA4OTD4JHOQANNB';
$FILE_TARGET_DIR = __DIR__ . '/'; // /home/digitala/public_html/insights/

// 2. --- AUTHORIZATION CHECK ---
$headers = getallheaders();
$auth_header = $headers['Authorization'] ?? '';

if (strpos($auth_header, 'Bearer ') !== 0) {
    http_response_code(401);
    die(json_encode(['error' => 'Authentication header missing.']));
}

$received_key = substr($auth_header, 7);
if ($received_key !== $EXPECTED_API_KEY) {
    http_response_code(403);
    die(json_encode(['error' => 'Invalid API Key. Access denied.']));
}

// 3. --- EXPECTED FILE TYPES ---
$allowed_extensions = ['csv', 'html', 'jpg', 'jpeg', 'json','png'];
$uploaded_files = [];

// Helper to process one file
function process_file($file, $target_dir, $allowed_exts)
{
    if ($file['error'] !== UPLOAD_ERR_OK) {
        return ['error' => 'Upload error: ' . $file['error']];
    }

    $file_name = basename($file['name']);
    $file_ext = strtolower(pathinfo($file_name, PATHINFO_EXTENSION));

    if (!in_array($file_ext, $allowed_exts)) {
        return ['error' => "Disallowed file type: {$file_ext}. Allowed: " . implode(', ', $allowed_exts)];
    }

    // Optional: enforce naming pattern (e.g., YYYYMMDD_*.ext)
    if (!preg_match('/^\d{8}_.*\.(csv|html|jpg|jpeg|json)$/i', $file_name)) {
        return ['error' => 'Filename must start with 8-digit date (e.g., 20251119_article.html)'];
    }

    $target_path = $target_dir . $file_name;
    if (move_uploaded_file($file['tmp_name'], $target_path)) {
        return ['success' => true, 'filename' => $file_name, 'path' => $target_path];
    } else {
        return ['error' => 'Failed to move uploaded file. Check permissions (755) and disk space.'];
    }
}

// 4. --- HANDLE MULTIPLE FILES ---
$results = [];

// Check for 'insights_file' (main content: CSV, HTML, or JSON)
if (isset($_FILES['insights_file'])) {
    $results[] = process_file($_FILES['insights_file'], $FILE_TARGET_DIR, $allowed_extensions);
}

// Check for 'insight_image' (thumbnail)
if (isset($_FILES['insight_image'])) {
    $img_result = process_file($_FILES['insight_image'], $FILE_TARGET_DIR, ['jpg', 'jpeg', 'png']);
    $results[] = $img_result;
}

// 5. --- FINAL RESPONSE ---
$successes = array_filter($results, fn($r) => $r['success'] ?? false);
$errors = array_filter($results, fn($r) => isset($r['error']));

if (!empty($errors)) {
    http_response_code(400);
    echo json_encode([
        'status' => 'partial_failure',
        'message' => 'Some files failed to upload.',
        'errors' => array_column($errors, 'error'),
        'saved_files' => array_column($successes, 'filename')
    ]);
} elseif (empty($successes)) {
    http_response_code(400);
    echo json_encode(['error' => 'No valid files received.']);
} else {
    http_response_code(200);
    echo json_encode([
        'status' => 'success',
        'message' => 'Files uploaded successfully.',
        'saved_files' => array_column($successes, 'filename')
    ]);
}
