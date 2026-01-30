const std = @import("std");
const root = @import("root.zig");

var M: usize = 1024;
var N: usize = 1024;
var P: usize = 1024;

var NUM_THREADS: usize = undefined;

const Schedule = root.Schedule;

const Matrix = struct {
    rows: usize,
    cols: usize,
    mat: []i32,

    pub fn init(al: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const buf = try al.alloc(i32, rows * cols);
        return Matrix{ .rows = rows, .cols = cols, .mat = buf };
    }

    pub fn deinit(self: *Matrix, al: std.mem.Allocator) void {
        al.free(self.mat);
        self.* = undefined;
    }

    pub fn at(self: *const Matrix, i: usize, j: usize) *i32 {
        return &self.mat[i * self.cols + j];
    }

    pub fn fill_random(self: *Matrix, rng: anytype) void {
        for (self.mat) |*elem| {
            elem.* = rng.intRangeAtMost(i32, -10, 10);
        }
    }

    pub fn print(self: *const Matrix) void {
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                std.debug.print("{d:4} ", .{self.at(i, j).*});
            }
            std.debug.print("\n", .{});
        }
    }
};

const Scheduler = struct {
    threads: []std.Thread,
    mat_A: *const Matrix,
    mat_B: *const Matrix,
    mat_C: Matrix,
    mode: Schedule,

    pub fn init(
        al: std.mem.Allocator,
        mat_A: *const Matrix,
        mat_B: *const Matrix,
        num_threads: usize,
        mode: Schedule,
    ) !Scheduler {
        const threads_buf = try al.alloc(std.Thread, num_threads);

        return Scheduler{
            .threads = threads_buf,
            .mat_A = mat_A,
            .mat_B = mat_B,
            .mat_C = try Matrix.init(al, mat_A.rows, mat_B.cols),
            .mode = mode,
        };
    }

    pub fn deinit(self: *Scheduler, al: std.mem.Allocator) void {
        al.free(self.threads);
        self.mat_C.deinit(al);
        self.* = undefined;
    }

    pub fn compute_matrix(self: *Scheduler) !void {
        switch (self.mode) {
            .sequential => self.compute_matrix_sequential(),
            .chunked => try self.compute_matrix_chunked(),
            .cyclic => try self.compute_matrix_cyclic(),
            .dynamic => try self.compute_matrix_dynamic(),
        }
    }

    pub fn compute_matrix_sequential(self: *Scheduler) void {
        for (0..self.mat_C.rows) |row| {
            compute_row(self.mat_A, self.mat_B, &self.mat_C, row);
        }
    }

    pub fn compute_matrix_chunked(self: *Scheduler) !void {
        const max_threads = @min(NUM_THREADS, self.mat_C.rows);
        const chunk_size = self.mat_C.rows / max_threads;

        for (0..max_threads) |id| {
            const start_row = id * chunk_size;
            const end_row = if (id == max_threads - 1)
                self.mat_C.rows
            else
                start_row + chunk_size;

            const thread = try std.Thread.spawn(.{}, compute_rows_chunked, .{
                self.mat_A,
                self.mat_B,
                &self.mat_C,
                start_row,
                end_row,
            });

            self.threads[id] = thread;
        }

        for (self.threads[0..max_threads]) |thread| {
            thread.join();
        }
    }

    pub fn compute_matrix_cyclic(self: *Scheduler) !void {
        const max_threads = @min(NUM_THREADS, self.mat_C.rows);
        for (0..max_threads) |id| {
            const thread = try std.Thread.spawn(.{}, compute_rows_cyclic, .{
                self.mat_A,
                self.mat_B,
                &self.mat_C,
                max_threads,
                id,
            });

            self.threads[id] = thread;
        }

        for (self.threads[0..max_threads]) |thread| {
            thread.join();
        }
    }

    pub fn compute_matrix_dynamic(self: *Scheduler) !void {
        const max_threads = @min(NUM_THREADS, self.mat_C.rows);
        var atomic_counter = std.atomic.Value(usize).init(0);

        for (0..max_threads) |id| {
            const thread = try std.Thread.spawn(.{}, compute_rows_dynamic, .{
                self.mat_A,
                self.mat_B,
                &self.mat_C,
                &atomic_counter,
            });

            self.threads[id] = thread;
        }

        for (self.threads[0..max_threads]) |thread| {
            thread.join();
        }
    }
};

pub fn compute_rows_chunked(
    mat_A: *const Matrix,
    mat_B: *const Matrix,
    mat_C: *Matrix,
    start_row: usize,
    end_row: usize,
) void {
    for (start_row..end_row) |row| {
        compute_row(mat_A, mat_B, mat_C, row);
    }
}

pub fn compute_rows_cyclic(
    mat_A: *const Matrix,
    mat_B: *const Matrix,
    mat_C: *Matrix,
    interval: usize,
    thread_id: usize,
) void {
    var row = thread_id;
    while (row < mat_A.rows) : (row += interval) {
        compute_row(mat_A, mat_B, mat_C, row);
    }
}

pub fn compute_rows_dynamic(
    mat_A: *const Matrix,
    mat_B: *const Matrix,
    mat_C: *Matrix,
    counter: *std.atomic.Value(usize),
) void {
    const num_rows = mat_C.rows;
    while (true) {
        const row = counter.fetchAdd(1, .monotonic);
        if (row >= num_rows) break;
        compute_row(mat_A, mat_B, mat_C, row);
    }
}

pub fn compute_row(
    mat_A: *const Matrix,
    mat_B: *const Matrix,
    mat_C: *Matrix,
    row: usize,
) void {
    for (0..mat_B.cols) |j| {
        var sum: i32 = 0;
        for (0..mat_A.cols) |k| {
            sum += mat_A.at(row, k).* * mat_B.at(j, k).*;
        }
        mat_C.at(row, j).* = sum;
    }
}

pub fn run(schedule: Schedule, m: usize, n: usize, p: usize, num_threads: usize) !void {
    NUM_THREADS = num_threads;
    M = m;
    N = n;
    P = p;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const al = gpa.allocator();

    const rng = std.crypto.random;

    var mat_A = try Matrix.init(al, M, N);
    defer mat_A.deinit(al);
    mat_A.fill_random(rng);

    var mat_B = try Matrix.init(al, N, P);
    defer mat_B.deinit(al);
    mat_B.fill_random(rng);

    var timer = try std.time.Timer.start();

    std.debug.print("\n---------EVALUATING---------\n", .{});
    var sched = try Scheduler.init(al, &mat_A, &mat_B, NUM_THREADS, schedule);
    defer sched.deinit(al);

    timer.reset();
    try sched.compute_matrix();
    const time = timer.read();

    std.debug.print(
        "\nTiming (ns): {d}",
        .{time},
    );
}
