import runtime


def handler(job):
    try:
        p = runtime.validate(job.get('input'))
        with runtime.LOCK:
            return runtime.execute(p)
    except Exception as error:
        detail = runtime.safe_text(error)
        print(type(error).__name__ + ': ' + detail, flush=True)
        return {'error': detail, 'error_type': type(error).__name__, 'worker_version': runtime.VERSION}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
