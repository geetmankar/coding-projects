use color_eyre::eyre::{Error, Result};
use kdam::tqdm;
use ndarray::{s, stack, Array1, Array3, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use ndarray_stats::QuantileExt;
use rustplotlib::{backend::Matplotlib, Axes2D, Backend, Figure, Line2D, Scatter};

macro_rules! energy_range {
    ($ke: expr, $pe: expr) => {
        (
            (stack(Axis(0), &[$ke, $pe])?.to_owned().min()?.to_owned()),
            (stack(Axis(0), &[$ke, $pe])?.to_owned().max()?.to_owned()),
        )
    };
}
//ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>
//ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>
//ArrayBase<ViewRepr<f64>, Dim<[usize; 3]>>
//ArrayBase<ViewRepr<f64>, Dim<[usize; 1]>>

pub fn plot_nbodysystem(
    pos: Array3<f64>,
    ke: Array1<f64>,
    pe: Array1<f64>,
    t_all: Array1<f64>,
    filename: String,
) -> Result<(), Error> {
    let e_range = energy_range!(ke.view(), pe.view());

    let tmax = t_all[t_all.len() - 1];

    for i in tqdm!(
        0..t_all.len(),
        desc = "Plotting...",
        colour = "green",
        unit = " frames"
    ) {
        let fname = filename.clone() + &format!("_{i}.png");

        let pos_interim = pos.slice(s![.., .., i]).clone().to_owned();
        let pos_x = pos_interim.slice(s![.., 0]).to_vec();
        let pos_y = pos_interim.slice(s![.., 1]).to_vec();
        // println!("xlen = {:?}; ylen = {:?}", &pos_x.len(), &pos_y.len());
        let ke_i = ke.slice(s![..=i]).clone().to_owned().into_raw_vec();
        let pe_i = pe.slice(s![..=i]).clone().to_owned().into_raw_vec();
        let t_all_i = t_all.slice(s![..=i]).to_owned().into_raw_vec();

        let ax1 = Axes2D::new()
            .add(Scatter::new("").data(&pos_x, &pos_y).marker("o"))
            .xlabel(r"$X$")
            .ylabel(r"$Y$")
            .xlim(-4.0, 4.0)
            .ylim(-4.0, 4.0);

        let ax2 = Axes2D::new()
            .add(
                Line2D::new(r"$K.E.$")
                    .data(&t_all_i, &ke_i)
                    .marker("o")
                    .linestyle("-")
                    .linewidth(1.0),
            )
            .add(
                Line2D::new(r"P.E.$")
                    .data(&t_all_i, &pe_i)
                    .color("red")
                    .marker("x")
                    .linestyle("-")
                    .linewidth(1.0),
            )
            .xlabel("Time")
            .ylabel("Energy")
            .legend("center right")
            .xlim(0., tmax.clone())
            .ylim(e_range.0, e_range.1);

        let fig = Figure::new().subplots(2, 1, vec![Some(ax1), Some(ax2)]);
        // ! FIX THE SIZES OF THE FIGURES
        let mut mpl = Matplotlib::new()?;
        // mpl.set_style("science")?;
        fig.apply(&mut mpl)?;
        // fig.apply(mpl)?;
        mpl.savefig(fname.as_str())?;
        mpl.wait()?;
    }

    Ok(())
}

// ----------------------------------------------------------------
