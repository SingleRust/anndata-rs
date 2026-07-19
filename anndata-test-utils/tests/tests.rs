use anndata::concat::{JoinType, concat};
use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, ArrayElemOp, Backend, Selectable};
use anndata_hdf5::H5;
use anndata_test_utils as utils;
use anndata_test_utils::with_tmp_dir;
use anndata_zarr::Zarr;
use sprs::CsMatI;

#[test]
fn test_basic() {
    utils::test_basic::<H5>();
    utils::test_basic::<Zarr>();
}

#[test]
fn test_complex_dataframe() {
    let input = "tests/data/sample.h5ad";
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata = AnnData::<H5>::open(H5::open(&input).unwrap()).unwrap();
        adata.write::<H5, _>(file, None, None).unwrap();
    });

    with_tmp_dir(|dir| {
        let file = dir.join("test.zarr");
        let adata = AnnData::<H5>::open(H5::open(&input).unwrap()).unwrap();
        adata.write::<Zarr, _>(file, None, None).unwrap();
    });
}

#[test]
fn test_mixed_layers() {
    utils::test_mixed_layers::<H5>();
    utils::test_mixed_layers::<Zarr>();
}

#[test]
fn test_pairwise() {
    utils::test_pairwise::<H5>();
    utils::test_pairwise::<Zarr>();
}

#[test]
fn test_sparse_edge_cases() {
    utils::test_sparse_edge_cases::<H5>();
    utils::test_sparse_edge_cases::<Zarr>();
}

#[test]
fn test_corrupt_sparse_full_read() {
    utils::test_corrupt_sparse_full_read::<H5>();
    utils::test_corrupt_sparse_full_read::<Zarr>();
}

#[test]
fn test_anndataset_mixed_layouts() {
    utils::test_anndataset_mixed_layouts::<H5>();
    utils::test_anndataset_mixed_layouts::<Zarr>();
}

#[test]
fn test_sparse_extraction_select() {
    utils::test_sparse_extraction_select::<H5>();
    utils::test_sparse_extraction_select::<Zarr>();
}

#[test]
fn test_parallel_reading_stress() {
    utils::test_parallel_reading_stress::<H5>();
    utils::test_parallel_reading_stress::<Zarr>();
}

#[test]
fn test_save() {
    utils::test_save::<H5>();
    utils::test_save::<Zarr>();
}

#[test]
fn test_speacial_cases() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_speacial_cases(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_speacial_cases(|| adata_gen());
    })
}

#[test]
fn test_noncanonical() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_noncanonical(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_noncanonical(|| adata_gen());
    })
}

#[test]
fn test_io() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_io(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_io(|| adata_gen());
    })
}

#[test]
fn test_index() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_index(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_index(|| adata_gen());
    })
}

#[test]
fn test_iterator() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_iterator(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_iterator(|| adata_gen());
    })
}

#[test]
fn test_concat_sparse_outer() {
    with_tmp_dir(|dir| {
        let adata1 = AnnData::<H5>::new(dir.join("input1.h5ad")).unwrap();
        let adata2 = AnnData::<H5>::new(dir.join("input2.h5ad")).unwrap();
        let output = AnnData::<H5>::new(dir.join("output.h5ad")).unwrap();

        let x1 = CsMatI::<i64, i64, u64>::new((2, 2), vec![0, 2, 3], vec![0, 1, 1], vec![1, 2, 3]);
        let x2 = CsMatI::<i64, i64, u64>::new((1, 2), vec![0, 2], vec![0, 1], vec![4, 5]);
        adata1.set_x(x1).unwrap();
        adata1
            .set_obs_names(vec!["o1".to_string(), "o2".to_string()].into())
            .unwrap();
        adata1
            .set_var_names(vec!["a".to_string(), "b".to_string()].into())
            .unwrap();
        adata2.set_x(x2).unwrap();
        adata2.set_obs_names(vec!["o3".to_string()].into()).unwrap();
        adata2
            .set_var_names(vec!["b".to_string(), "c".to_string()].into())
            .unwrap();

        concat::<_, _, String>(&[adata1, adata2], JoinType::Outer, None, None, &output).unwrap();

        let expected = CsMatI::<i64, i64, u64>::new(
            (3, 3),
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 1, 2],
            vec![1, 2, 3, 4, 5],
        );
        assert_eq!(
            output.x().get::<CsMatI<i64, i64, u64>>().unwrap().unwrap(),
            expected
        );
    });
}

#[test]
fn test_split_sparse() {
    with_tmp_dir(|dir| {
        let adata = AnnData::<H5>::new(dir.join("input.h5ad")).unwrap();
        let x = CsMatI::<i64, i64, u64>::new(
            (5, 3),
            vec![0, 2, 3, 4, 6, 7],
            vec![0, 2, 1, 2, 0, 1, 2],
            vec![1, 2, 3, 4, 5, 6, 7],
        );
        adata.set_x(x.clone()).unwrap();
        adata
            .set_obs_names((0..5).map(|x| x.to_string()).collect())
            .unwrap();
        adata
            .set_var_names((0..3).map(|x| x.to_string()).collect())
            .unwrap();

        let keys = ["A", "A", "B", "A", "B"].map(|key| Some(key.to_string()));
        let split = adata
            .split_obs_by::<H5, _>(&keys, dir.join("split"))
            .unwrap();
        let expected_a = x.select(&[SelectInfoElem::from(vec![0, 1, 3]), SelectInfoElem::full()]);

        assert_eq!(
            split["A"]
                .x()
                .get::<CsMatI<i64, i64, u64>>()
                .unwrap()
                .unwrap(),
            expected_a
        );
    });
}

#[test]
fn test_take_x() {
    utils::test_take_x::<H5>();
    utils::test_take_x::<Zarr>();
}

#[test]
fn test_obsm_drain() {
    utils::test_obsm_drain::<H5>();
    utils::test_obsm_drain::<Zarr>();
}

#[test]
fn test_backend_interop() {
    utils::test_backend_interop::<H5, Zarr>();
    utils::test_backend_interop::<Zarr, H5>();
}

#[test]
fn test_uns_nesting() {
    utils::test_uns_nesting::<H5>();
    utils::test_uns_nesting::<Zarr>();
}
