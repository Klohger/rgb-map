#![feature(new_uninit)]
#![feature(maybe_uninit_uninit_array_transpose)]
use core::{num, slice};
use std::{
    cmp::Ordering,
    fmt::Display,
    fs::File,
    io::{stdout, BufReader, BufWriter, Read, Write},
    mem::{self, MaybeUninit},
    os::windows::fs::MetadataExt,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use oklab::{srgb_to_oklab, Oklab, RGB};
use png::ColorType;

fn get_r8g8b8(i: usize) -> (u8, u8, u8) {
    (
        ((i) & ((1 << 8) - 1)) as u8,
        ((i >> 8) & ((1 << 8) - 1)) as u8,
        ((i >> 16) & ((1 << 8) - 1)) as u8,
    )
}
fn get_i_from_r8g8b8(rgb: &[u8; 3]) -> usize {
    (rgb[0] as usize) + ((rgb[1] as usize) << 8) + ((rgb[2] as usize) << 16)
}
fn r8g8b8_to_oklab(rgb: (u8, u8, u8)) -> Oklab {
    let (r, g, b) = rgb;
    srgb_to_oklab(RGB { r, g, b })
}
fn get_r3g3b2(i: u8) -> [u8; 3] {
    [(i) & 0b111, (i >> 3) & 0b111, (i >> 6)]
}
fn r3g3b2_to_oklab(rgb: [u8; 3]) -> Oklab {
    let [r, g, b] = rgb;
    srgb_to_oklab(RGB {
        r: ((r as usize * ((1 << 8) - 1)) / ((1 << 3) - 1)) as u8,
        g: ((g as usize * ((1 << 8) - 1)) / ((1 << 3) - 1)) as u8,
        b: ((b as usize * ((1 << 8) - 1)) / ((1 << 2) - 1)) as u8,
    })
}
fn mix_oklab<const N: usize>(colors: [Oklab; N]) -> Oklab {
    let mut sum = colors.into_iter().fold(
        Oklab {
            l: 0.0,
            a: 0.0,
            b: 0.0,
        },
        |mut sum, color| {
            sum.l += color.l;
            sum.a += color.a;
            sum.b += color.b;
            sum
        },
    );
    sum.l /= N as f32;
    sum.a /= N as f32;
    sum.b /= N as f32;
    sum
}
fn oklab_distance(mut lhs: Oklab, rhs: &Oklab) -> f32 {
    lhs.l -= rhs.l;
    lhs.a -= rhs.a;
    lhs.b -= rhs.b;
    (lhs.l.mul_add(lhs.l, lhs.a.mul_add(lhs.a, lhs.b * lhs.b))).sqrt()
}
/*
const fn get_compare_length<const N: usize>() -> usize {
    ((N - 0) * (N - 1) * (N - 2) * (N - 3)) / 24
}
*/

// |length : usize, num_comparators : usize| -> usize
//     (0..num_comparators)
//         .map(|n| (length + n) )
//         .reduce(|a,b| { a * b }) /
//     (1..=num_comparators).reduce(|a,b| a*b)
const fn calculate_unique_combination_length(count: usize, base: usize) -> usize {
    let dividend = {
        let mut n = 0;
        let mut product = 1;
        while n < base {
            product *= count + n;
            n += 1;
        }
        product
    };
    let divisor = {
        let mut n = 2;
        let mut product = 1;
        while n <= base {
            product *= n;
            n += 1;
        }
        product
    };
    dividend / divisor
}
const R3G3B2_MIX_TABLE_LEN: usize = (256 * (256 + 1)) / 2;

fn generate_function<T, S: Display, F: Fn(Box<[MaybeUninit<T>]>) -> Box<[T]>, P: AsRef<Path>>(
    size: usize,
    name: S,
    file_path: P,
    function: F,
) -> Box<[T]> {
    println!("Generating {name}");
    let path = file_path.as_ref();
    let mut arr = Box::new_uninit_slice(size);
    match path.metadata() {
        Ok(metadata)
            if metadata.is_file()
                && metadata.file_size() == arr.len() as u64 * mem::size_of::<T>() as u64 =>
        {
            let mut file = BufReader::new(File::open(path).unwrap());
            file.read_exact(unsafe {
                slice::from_raw_parts_mut(
                    arr.as_mut_ptr() as *mut _,
                    arr.len() * mem::size_of::<T>(),
                )
            })
            .expect("whoopsie filey readie go no no on");
            let arr = unsafe { arr.assume_init() };
            arr
        }
        _ => {
            let arr = function(arr);
            let mut file = BufWriter::new(File::create(path).unwrap());
            file.write_all(unsafe {
                slice::from_raw_parts(arr.as_ptr() as *const _, arr.len() * mem::size_of::<T>())
            })
            .expect("whoopsie filey writey go no no no");
            arr
        }
    }
}

fn generate_r8g8b8_oklab_map() -> Box<[Oklab]> {
    generate_function(
        256 * 256 * 256,
        "\"r8g8b8 -> oklab\" map",
        "r8g8b8_oklab_map.cache",
        |mut arr| {
            for i in 0..arr.len() {
                let rgb = get_r8g8b8(i);
                arr[i].write(r8g8b8_to_oklab(rgb));
            }
            let arr = unsafe { arr.assume_init() };
            arr
        },
    )
}
fn generate_r3g3b2_oklab_map() -> Box<[Oklab]> {
    generate_function(
        8 * 8 * 4,
        "\"r3g3b2 -> oklab\" map",
        "r3g3b2_oklab_map.cache",
        |mut arr| {
            for i in 0..arr.len() {
                let rgb = get_r3g3b2(i as u8);
                arr[i].write(r3g3b2_to_oklab(rgb));
            }
            let arr = unsafe { arr.assume_init() };
            arr
        },
    )
}

fn generate_r3g3b2_mix_table() -> Box<[[u8; 2]]> {
    generate_function(
        R3G3B2_MIX_TABLE_LEN,
        "r3g3b2 mix table",
        "r3g3b2_mix_table.cache",
        |mut arr| {
            let mut i = 0;
            for a in 0..256 {
                for b in (a)..256 {
                    arr[i].write([a as u8, b as u8]);
                    i += 1;
                }
            }
            assert!(i == R3G3B2_MIX_TABLE_LEN);
            let arr = unsafe { arr.assume_init() };
            arr
        },
    )
}
fn generate_mixed_r3g3b2_table(
    r3g3b3_mix_table: &[[u8; 2]],
    r3g3b2_oklab_map: &[Oklab],
) -> Box<[Oklab]> {
    generate_function(
        R3G3B2_MIX_TABLE_LEN,
        "mixed r3g3b2 table",
        "mixed_r3g3b2_table.cache",
        |mut arr| {
            for i in 0..arr.len() {
                let mixed_color =
                    mix_oklab(r3g3b3_mix_table[i].map(|el| r3g3b2_oklab_map[el as usize]));
                arr[i].write(mixed_color);
            }
            let arr = unsafe { arr.assume_init() };
            arr
        },
    )
}
/*
fn generate_distance(
    r8g8b8_oklab_map: Box<[Oklab]>,
    mixed_r3g3b2_table: Box<[Oklab]>,
) -> Box<[Box<[f32]>]> {
}
*/
fn generate_color_map(r8g8b8_oklab_map: &[Oklab], mixed_r3g3b2_table: &[Oklab]) -> Box<[usize]> {
    generate_function(
        256 * 256 * 256,
        "color map",
        "color_map.cache",
        move |mut arr| {
            let last_write = Arc::new(Mutex::new((
                Instant::now(),
                Duration::ZERO,
                0usize,
                Instant::now(),
                stdout(),
            )));
            let arr = rayon::in_place_scope(move |scope| {
                for r8g8b8_i in 0..arr.len() {
                    scope.spawn({
                        let location =
                            unsafe { &mut *(&mut arr[r8g8b8_i] as *mut MaybeUninit<usize>) };
                        let last_write = last_write.clone();
                        move |_| {
                            let (closest, _) = unsafe {
                                (0..mixed_r3g3b2_table.len())
                                    .map(|mixed_color_i| {
                                        let distance = oklab_distance(
                                            mixed_r3g3b2_table[mixed_color_i],
                                            &r8g8b8_oklab_map[r8g8b8_i],
                                        );
                                        (mixed_color_i, distance)
                                    })
                                    .reduce(|a, b| match a.1.partial_cmp(&b.1) {
                                        None => unreachable!(),
                                        Some(ordering) => match ordering {
                                            Ordering::Less => a,
                                            Ordering::Equal | Ordering::Greater => b,
                                        },
                                    })
                                    .unwrap_unchecked()
                            };

                            let mut last_write = last_write.lock().unwrap();
                            let write = Instant::now();
                            let time_taken = write - last_write.0;
                            last_write.1 += time_taken;
                            last_write.0 = write;
                            last_write.2 += 1;
                            if (write - last_write.3) > Duration::from_secs(1) {
                                last_write.3 = write;
                                let time_left = (last_write.1 / (last_write.2 as u32))
                                    * ((256 * 256 * 256) - (last_write.2)) as u32;
                                let time = time_left.as_secs();
                                last_write
                                    .4
                                    .write_all(
                                        format!(
                                            "{}y {}w {}d {}h {}m {}s left\n",
                                            time / 31536000,
                                            (time / 604800) % 604800,
                                            (time / 86400) % 86400,
                                            (time / 3600) % 3600,
                                            (time / 60) % 60,
                                            time % 60
                                        )
                                        .as_bytes(),
                                    )
                                    .unwrap();
                            }
                            drop(last_write);
                            location.write(closest);
                        }
                    });
                }
                arr
            });

            let arr = unsafe { arr.assume_init() };
            arr
        },
    )
}
fn generate_final_color_map(color_map: &[usize], r3g3b2_mix_table: &[[u8; 2]]) -> Box<[[u8; 2]]> {
    generate_function(
        color_map.len(),
        "final color map!",
        "color_map_final.cache",
        |mut arr| {
            for i in 0..arr.len() {
                arr[i].write(r3g3b2_mix_table[color_map[i]]);
            }
            unsafe { arr.assume_init() }
        },
    )
}
fn main() {
    println!("{}", calculate_unique_combination_length(256, 4) as f32 / calculate_unique_combination_length(256, 2) as f32)
    /*
    let r8g8b8_oklab_map = generate_r8g8b8_oklab_map();

    let r3g3b2_oklab_map = generate_r3g3b2_oklab_map();

    let r3g3b2_mix_table = generate_r3g3b2_mix_table();

    let mixed_r3g3b2_table = generate_mixed_r3g3b2_table(&r3g3b2_mix_table, &r3g3b2_oklab_map);
    drop(r3g3b2_oklab_map);

    let color_map = generate_color_map(&r8g8b8_oklab_map, &mixed_r3g3b2_table);
    drop(r8g8b8_oklab_map);
    drop(mixed_r3g3b2_table);

    let color_map = generate_final_color_map(&color_map, &r3g3b2_mix_table);
    drop(r3g3b2_mix_table);

    apply_color_map_to_image("input.png", "output.png", &color_map);
    drop(color_map);
    */
}

fn apply_color_map_to_image(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    color_map: &[[u8; 2]],
) {
    println!("applying color map to image");
    let input = png::Decoder::new(BufReader::new(File::open(input_path).unwrap()));
    let mut input = input.read_info().unwrap();
    if input.info().color_type != ColorType::Rgb {
        panic!(
            "Excpected {:?}, got {:?}",
            ColorType::Rgb,
            input.info().color_type
        )
    }
    let width = input.info().width as usize;
    let height = input.info().height as usize;
    let mut buf = vec![0; width * height * 3];
    input.next_frame(&mut buf).unwrap();
    drop(input);
    for (i, a) in buf.chunks_mut(3).enumerate() {
        let checker = (i + (i / width)) & 1;

        let rgb_i = get_i_from_r8g8b8((&*a).try_into().unwrap());

        let mut color = get_r3g3b2(color_map[rgb_i][checker]);

        color[0] = (((color[0] as usize) * 255) / 7) as u8;
        color[1] = (((color[1] as usize) * 255) / 7) as u8;
        color[2] = (((color[2] as usize) * 255) / 3) as u8;
        a.copy_from_slice(&color);
    }
    let mut image = png::Encoder::new(
        BufWriter::new(File::create(output_path).unwrap()),
        width as u32,
        height as u32,
    );
    image.set_color(ColorType::Rgb);
    let mut image = image.write_header().unwrap();
    image.write_image_data(&buf).unwrap();
}
